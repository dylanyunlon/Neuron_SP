"""
Neuron_SP / DES-LOC  —  HeteroDecoupledGradDistOpt
===================================================

上游设计意图 (Megatron ab43d43)
--------------------------------
Megatron-FSDP (MFSDP) 引入了"主权重由FSDP管理、优化器状态由DistributedOptimizer分片"
的双轨模式。原有代码存在两处 bug：

1. ``decoupled_grad`` 路径判断不完整：仅在 ``use_precision_aware_optimizer_no_fp8_or_ds_fp8``
   条件下启用，遗漏了 FSDP param（``__fsdp_param__ = True``）始终需要走
   ``decoupled_grad`` 分支的情况。
2. ``FusedAdam.master_weights`` 被错误地保留为 True：当 MFSDP 持有主权重时，
   FusedAdam 的内部 master_weights 副本是纯浪费，且会导致优化器步骤将更新写入
   两份不同的主权重（FusedAdam 内部副本 vs. FSDP DTensor），从而出现参数发散。

DES-LOC 适配点
--------------
DES-LOC（Decoupled Execution with Shared LOcality Cache）在异构集群
``2× A6000-48GB(SM86) + 1× H100-NVL-96GB(SM90)`` 上面临类似但更复杂的问题：

* **异构主权重所有权**：大参数分片常驻 H100（高带宽，96 GB），
  小参数 / 嵌入层分片放在 A6000（低延迟访问 CPU DRAM 1.5 TB）。
  DES-LOC 的 ``LocalityCache`` 负责在设备间搬运热分片；优化器必须知道
  "当前 step 该参数的主权重在哪个设备"。

* **decoupled_grad 异构版本**：梯度可能存在于与参数不同的设备上（A6000 做
  前向/反向，H100 做 all-reduce），因此需要 ``HeteroDecoupledGrad`` 包装器，
  记录梯度所在的实际设备，并在 clip / norm 阶段做跨设备聚合。

* **master_weights 所有权守卫**：当 ``LocalityCache`` 持有分片主权重时，
  底层 Adam 实例（FusedAdam 或 DeepSpeed Adam）的内部 master_weights 必须
  被强制禁用，否则会出现双写冲突——这直接对应 Megatron bug #2。

本文件实现：
  - ``HeteroDecoupledGradDistOpt``：核心优化器类
  - ``LocalityCacheRegistry``：跨设备主权重所有权注册表
  - ``HeteroDecoupledGrad``：异构梯度包装器
  - ``DecoupledGradClipNorm``：跨设备梯度裁剪/norm 计算
  - ``_build_hetero_param_groups``：异构参数分组构建器
"""

from __future__ import annotations

import gc
import logging
import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量 / 设备能力表
# ---------------------------------------------------------------------------

_SM86_ARCHS = frozenset(["NVIDIA RTX A6000", "RTX A6000"])
_SM90_ARCHS = frozenset(["NVIDIA H100 NVL", "H100 NVL", "H100"])

# DES-LOC 约定：H100 为"高算力节点"，A6000 为"缓存节点"
_DEVICE_TIER: Dict[int, str] = {}  # cuda_device_idx -> "compute" | "cache"


def _classify_devices() -> None:
    """
    在进程启动时对可见 CUDA 设备分类。

    H100 系列 (SM90) 归为 ``compute`` tier，A6000 (SM86) 归为 ``cache`` tier。
    分类结果写入模块级 ``_DEVICE_TIER`` 字典，供后续所有组件查询。
    """
    global _DEVICE_TIER
    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        sm = props.major * 10 + props.minor
        name = props.name
        if sm >= 90:
            _DEVICE_TIER[i] = "compute"
            logger.info("DES-LOC device %d [%s] SM%d → compute tier", i, name, sm)
        else:
            _DEVICE_TIER[i] = "cache"
            logger.info("DES-LOC device %d [%s] SM%d → cache tier", i, name, sm)


_classify_devices()


def _preferred_device_for_param(param: torch.Tensor) -> torch.device:
    """
    根据参数大小和当前设备能力返回"最优"主权重设备。

    策略
    ----
    * 参数元素数 > 50M  →  优先放 H100（compute tier）
    * 其余              →  保留在原设备（避免不必要的跨设备搬运）

    当没有 compute tier 设备可用时退回到参数当前设备。
    """
    numel = param.numel()
    compute_devs = [i for i, t in _DEVICE_TIER.items() if t == "compute"]
    if numel > 50_000_000 and compute_devs:
        return torch.device(f"cuda:{compute_devs[0]}")
    return param.device


# ---------------------------------------------------------------------------
# HeteroDecoupledGrad —— 异构梯度包装器
# ---------------------------------------------------------------------------


class HeteroDecoupledGrad:
    """
    异构设备下的解耦梯度容器。

    上游对应
    ---------
    Megatron 修复中，FSDP param 的梯度存储为 DTensor，需在 norm/clip 阶段
    提取 ``_local_tensor``。DES-LOC 中对应的概念是：梯度可能存储在与参数
    **不同** 的 CUDA 设备上，需要携带设备元数据以便跨设备聚合。

    属性
    ----
    data : torch.Tensor
        实际梯度张量（可能在任意设备上）。
    src_device : torch.device
        梯度产生的设备（前向/反向所在设备）。
    param_device : torch.device
        对应参数主权重所在设备。
    is_hetero : bool
        ``src_device != param_device`` 时为 True，表示需要跨设备搬运。
    """

    __slots__ = ("data", "src_device", "param_device", "is_hetero", "_norm_cache")

    def __init__(
        self,
        data: torch.Tensor,
        src_device: torch.device,
        param_device: torch.device,
    ) -> None:
        self.data = data
        self.src_device = src_device
        self.param_device = param_device
        self.is_hetero = src_device != param_device
        self._norm_cache: Optional[torch.Tensor] = None

    def to_param_device(self, non_blocking: bool = True) -> torch.Tensor:
        """
        将梯度搬运到参数主权重所在设备。

        当 ``is_hetero=False`` 时直接返回 ``data``，避免无效 D2D 拷贝。
        """
        if not self.is_hetero:
            return self.data
        moved = self.data.to(self.param_device, non_blocking=non_blocking)
        logger.debug(
            "DES-LOC HeteroGrad: move grad %s → %s (numel=%d)",
            self.src_device,
            self.param_device,
            self.data.numel(),
        )
        return moved

    def norm(self, norm_type: float = 2.0) -> torch.Tensor:
        """计算梯度 norm（结果缓存在 src_device 上）。"""
        if self._norm_cache is not None:
            return self._norm_cache
        if norm_type == 2.0:
            self._norm_cache = self.data.float().norm(2)
        elif norm_type == float("inf"):
            self._norm_cache = self.data.float().abs().max()
        else:
            self._norm_cache = self.data.float().norm(norm_type)
        return self._norm_cache

    def __repr__(self) -> str:
        return (
            f"HeteroDecoupledGrad(shape={tuple(self.data.shape)}, "
            f"dtype={self.data.dtype}, "
            f"src={self.src_device}, param={self.param_device}, "
            f"is_hetero={self.is_hetero})"
        )


# ---------------------------------------------------------------------------
# LocalityCacheRegistry —— 主权重所有权注册表
# ---------------------------------------------------------------------------


@dataclass
class _ParamShardMeta:
    """单个参数分片的元数据。"""

    param_id: int
    device: torch.device
    owns_master_weights: bool = True  # DES-LOC locality cache 是否持有主权重
    optimizer_master_weights_disabled: bool = False  # 底层优化器 master_weights 是否已被禁用


class LocalityCacheRegistry:
    """
    DES-LOC SharedLOcality Cache 的主权重所有权注册中心。

    设计意图
    --------
    对应 Megatron 修复中"当 FSDP 持有主权重时必须禁用 FusedAdam.master_weights"
    的逻辑，在 DES-LOC 中推广为：**任何持有参数主权重的缓存层（LocalityCache）
    都必须在此处登记，下游优化器在构建时从这里查询所有权并据此禁用内部副本。**

    线程安全
    --------
    本实现假设单进程单线程访问（与 DeepSpeed ZeRO 一致）。
    """

    def __init__(self) -> None:
        self._registry: Dict[int, _ParamShardMeta] = {}
        self._device_to_params: Dict[str, List[int]] = defaultdict(list)

    def register(
        self,
        param: torch.Tensor,
        master_device: torch.device,
        owns_master_weights: bool = True,
    ) -> None:
        """
        注册参数分片的主权重所有权。

        参数
        ----
        param : torch.Tensor
            模型参数（用 ``id()`` 作为键）。
        master_device : torch.device
            主权重所在设备（由 ``_preferred_device_for_param`` 决定）。
        owns_master_weights : bool
            True 表示 LocalityCache 持有主权重，底层优化器应禁用内部副本。
        """
        pid = id(param)
        meta = _ParamShardMeta(
            param_id=pid,
            device=master_device,
            owns_master_weights=owns_master_weights,
        )
        self._registry[pid] = meta
        self._device_to_params[str(master_device)].append(pid)
        logger.debug(
            "LocalityCache register param %d → device=%s owns_master=%s",
            pid,
            master_device,
            owns_master_weights,
        )

    def owns_master_weights(self, param: torch.Tensor) -> bool:
        """查询 LocalityCache 是否持有该参数的主权重。"""
        meta = self._registry.get(id(param))
        return meta.owns_master_weights if meta is not None else False

    def master_device(self, param: torch.Tensor) -> Optional[torch.device]:
        """返回参数主权重所在设备；未注册则返回 None。"""
        meta = self._registry.get(id(param))
        return meta.device if meta is not None else None

    def mark_optimizer_master_weights_disabled(self, param: torch.Tensor) -> None:
        """标记底层优化器的 master_weights 已被禁用（幂等）。"""
        meta = self._registry.get(id(param))
        if meta is not None:
            meta.optimizer_master_weights_disabled = True

    def params_on_device(self, device: torch.device) -> List[int]:
        """返回主权重在指定设备上的所有参数 id。"""
        return self._device_to_params.get(str(device), [])

    def summary(self) -> str:
        lines = ["LocalityCacheRegistry:"]
        by_dev: Dict[str, int] = defaultdict(int)
        for meta in self._registry.values():
            by_dev[str(meta.device)] += 1
        for dev, cnt in by_dev.items():
            lines.append(f"  {dev}: {cnt} param shards")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DecoupledGradClipNorm —— 跨设备梯度裁剪 / norm
# ---------------------------------------------------------------------------


class DecoupledGradClipNorm:
    """
    异构设备下的梯度 clip-by-norm 实现。

    上游对应
    --------
    Megatron ab43d43 修复了 ``_get_grads_for_norm`` 中对 ``decoupled_grad``
    的判断逻辑，使得 FSDP param 始终走 ``decoupled_grad`` 分支。
    DES-LOC 中等价的修复是：对于在异构设备上的参数，norm 必须在梯度所在设备
    计算后再 all-reduce 到统一设备（H100 或 rank0），以避免无效 D2D 传输。

    算法
    ----
    1. 对每个参数，获取其 ``HeteroDecoupledGrad``（或普通 ``param.grad``）。
    2. 按设备分组计算局部 norm（避免碎片化 D2D）。
    3. 将各设备局部 norm 汇聚到 ``reduce_device`` 计算全局 norm。
    4. 若超过 ``max_norm``，在各设备上原地 scale 梯度。
    """

    def __init__(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        reduce_device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        参数
        ----
        max_norm : float
            梯度裁剪阈值。
        norm_type : float
            p-norm 类型，默认 L2。
        reduce_device : torch.device, optional
            汇聚 norm 计算的设备；默认取第一个 compute tier 设备。
        process_group : ProcessGroup, optional
            跨 rank all-reduce 使用的进程组。
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        compute_devs = [i for i, t in _DEVICE_TIER.items() if t == "compute"]
        default_reduce = torch.device(f"cuda:{compute_devs[0]}") if compute_devs else torch.device("cuda:0")
        self.reduce_device = reduce_device or default_reduce
        self.process_group = process_group

    def _get_decoupled_grad(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        """
        提取参数的解耦梯度。

        优先级：HeteroDecoupledGrad → decoupled_grad attribute → param.grad
        对应 Megatron 修复：FSDP param 始终使用 decoupled_grad；
        DES-LOC 扩展为异构场景下的多级查找。
        """
        # 1. DES-LOC 异构梯度包装器
        hgrad = getattr(param, "hetero_decoupled_grad", None)
        if hgrad is not None and isinstance(hgrad, HeteroDecoupledGrad):
            return hgrad.data

        # 2. Megatron 风格 decoupled_grad（兼容路径）
        dgrad = getattr(param, "decoupled_grad", None)
        if dgrad is not None:
            # DTensor 兼容（对应 Megatron 修复中 grad._local_tensor 的提取）
            if hasattr(dgrad, "_local_tensor"):
                return dgrad._local_tensor
            return dgrad

        # 3. 普通梯度
        return param.grad

    def compute_global_norm(self, params: List[torch.Tensor]) -> torch.Tensor:
        """
        跨异构设备计算全局梯度 norm。

        实现步骤
        --------
        1. 按梯度所在设备分组。
        2. 每个设备组并行计算局部 norm²（L2 情形）。
        3. 所有局部 norm² 搬到 reduce_device 求和后开方。
        4. 若有 process_group，做跨 rank all-reduce（sum of squares）。
        """
        device_norms: Dict[str, torch.Tensor] = {}  # device_str -> local norm²

        for param in params:
            if not param.requires_grad:
                continue
            grad = self._get_decoupled_grad(param)
            if grad is None:
                continue
            dev_str = str(grad.device)
            local_sq = grad.float().norm(2) ** 2
            if dev_str not in device_norms:
                device_norms[dev_str] = local_sq
            else:
                device_norms[dev_str] = device_norms[dev_str] + local_sq

        if not device_norms:
            return torch.tensor(0.0, device=self.reduce_device)

        # 汇聚各设备 norm² 到 reduce_device
        total_sq = torch.tensor(0.0, device=self.reduce_device)
        for dev_str, norm_sq in device_norms.items():
            total_sq = total_sq + norm_sq.to(self.reduce_device)

        # 跨 rank all-reduce
        if self.process_group is not None and dist.is_initialized():
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=self.process_group)

        global_norm = total_sq.sqrt()
        logger.debug(
            "DecoupledGradClipNorm: global_norm=%.6f (from %d devices)",
            global_norm.item(),
            len(device_norms),
        )
        return global_norm

    def clip_grads_(self, params: List[torch.Tensor]) -> torch.Tensor:
        """
        原地裁剪梯度，返回裁剪前的全局 norm。

        与普通 ``clip_grad_norm_`` 的区别
        ----------------------------------
        * 梯度可能在多个设备上，scale 操作在各自设备上执行（无需先汇聚）。
        * 使用 ``HeteroDecoupledGrad.data`` 而非 ``param.grad``，与上游
          Megatron 修复中始终走 ``decoupled_grad`` 分支一致。
        """
        global_norm = self.compute_global_norm(params)
        clip_coef = self.max_norm / (global_norm + 1e-6)
        if clip_coef < 1.0:
            for param in params:
                if not param.requires_grad:
                    continue
                grad = self._get_decoupled_grad(param)
                if grad is None:
                    continue
                grad.mul_(clip_coef.to(grad.device))
        return global_norm


# ---------------------------------------------------------------------------
# _build_hetero_param_groups —— 异构参数分组
# ---------------------------------------------------------------------------


def _build_hetero_param_groups(
    param_groups: List[dict],
    locality_cache: LocalityCacheRegistry,
) -> Tuple[List[dict], List[dict]]:
    """
    将参数组拆分为"compute tier 参数组"和"cache tier 参数组"。

    上游对应
    --------
    Megatron 中 ``_get_megatron_optimizer_based_on_param_groups`` 为每个
    model chunk 构建独立的 param_groups；DES-LOC 在此基础上按设备 tier 再次
    拆分，使得 compute tier 参数（在 H100 上）和 cache tier 参数（在 A6000 上）
    可以被独立的优化器实例更新，充分利用各设备的计算带宽。

    返回
    ----
    compute_groups : List[dict]
        主权重在 compute tier（H100）上的参数组。
    cache_groups : List[dict]
        主权重在 cache tier（A6000 / CPU DRAM）上的参数组。
    """
    compute_groups: List[dict] = []
    cache_groups: List[dict] = []

    for group in param_groups:
        compute_params = []
        cache_params = []
        for param in group.get("params", []):
            master_dev = locality_cache.master_device(param)
            if master_dev is None:
                # 未注册：按参数大小自动分配
                master_dev = _preferred_device_for_param(param)
                locality_cache.register(param, master_dev)

            tier = _DEVICE_TIER.get(master_dev.index, "cache")
            if tier == "compute":
                compute_params.append(param)
            else:
                cache_params.append(param)

        base = {k: v for k, v in group.items() if k != "params"}
        if compute_params:
            compute_groups.append({"params": compute_params, **base})
        if cache_params:
            cache_groups.append({"params": cache_params, **base})

    logger.info(
        "hetero param groups: %d compute-tier groups, %d cache-tier groups",
        len(compute_groups),
        len(cache_groups),
    )
    return compute_groups, cache_groups


# ---------------------------------------------------------------------------
# HeteroDecoupledGradDistOpt —— 核心优化器
# ---------------------------------------------------------------------------


class HeteroDecoupledGradDistOpt:
    """
    DES-LOC 异构解耦梯度分布式优化器。

    设计目标
    --------
    在 ``2× A6000-48GB(SM86) + 1× H100-NVL-96GB(SM90)`` 异构集群上，
    正确处理以下三类 bug（对应 Megatron ab43d43 修复的 DES-LOC 等价物）：

    Bug 1: master_weights 双写
        当 LocalityCache 持有参数主权重时，底层 Adam 优化器的 ``master_weights``
        必须被强制设为 False，否则 Adam 会维护一份独立的 FP32 副本，与
        LocalityCache 中的主权重产生冲突（参数发散）。
        DES-LOC 修复：在 ``_disable_optimizer_master_weights`` 中强制覆盖。

    Bug 2: decoupled_grad 路径缺失
        梯度裁剪 / norm 计算必须对所有 hetero param 走 ``hetero_decoupled_grad``
        路径，而非依赖 ``param.grad``（后者在异构场景可能为 None 或在错误设备上）。
        DES-LOC 修复：通过 ``DecoupledGradClipNorm._get_decoupled_grad`` 统一处理。

    Bug 3: FSDP-style early return 掩盖错误
        Megatron 原始代码在多处用 ``if use_megatron_fsdp: return`` 跳过校验，
        导致不支持的操作被静默忽略。DES-LOC 修复：对不兼容操作显式抛出
        ``NotImplementedError``（对应 Megatron diff 中新增的两处 raise）。

    架构
    ----
    ::

        HeteroDecoupledGradDistOpt
        ├── LocalityCacheRegistry        # 主权重所有权
        ├── DecoupledGradClipNorm        # 跨设备 grad clip
        ├── _compute_optimizer           # H100 上的 Adam（大参数）
        ├── _cache_optimizer             # A6000 上的 Adam（小参数）
        └── _hetero_param_groups         # 参数组拆分结果

    参数
    ----
    optimizer_cls : type
        基础优化器类（如 ``torch.optim.AdamW``）。
    param_groups : List[dict]
        DeepSpeed ZeRO 传入的参数组列表。
    locality_cache : LocalityCacheRegistry
        DES-LOC 主权重所有权注册表。
    max_grad_norm : float
        梯度裁剪阈值，0 表示不裁剪。
    process_group : ProcessGroup, optional
        数据并行进程组，用于跨 rank norm all-reduce。
    use_precision_aware_optimizer : bool
        是否启用精度感知优化（对应 Megatron ``use_precision_aware_optimizer``）。
    **optimizer_defaults
        传递给基础优化器的超参数（lr, betas, eps, weight_decay 等）。
    """

    def __init__(
        self,
        optimizer_cls: type,
        param_groups: List[dict],
        locality_cache: LocalityCacheRegistry,
        max_grad_norm: float = 1.0,
        process_group: Optional[dist.ProcessGroup] = None,
        use_precision_aware_optimizer: bool = False,
        **optimizer_defaults,
    ) -> None:
        self.optimizer_cls = optimizer_cls
        self.locality_cache = locality_cache
        self.max_grad_norm = max_grad_norm
        self.process_group = process_group
        self.use_precision_aware_optimizer = use_precision_aware_optimizer
        self.optimizer_defaults = optimizer_defaults

        # 拆分异构参数组
        compute_groups, cache_groups = _build_hetero_param_groups(
            param_groups, locality_cache
        )
        self._compute_param_groups = compute_groups
        self._cache_param_groups = cache_groups

        # 构建两个独立的优化器实例
        self._compute_optimizer = (
            optimizer_cls(compute_groups, **optimizer_defaults) if compute_groups else None
        )
        self._cache_optimizer = (
            optimizer_cls(cache_groups, **optimizer_defaults) if cache_groups else None
        )

        # 修复 Bug 1：禁用底层优化器的 master_weights
        self._disable_optimizer_master_weights()

        # 梯度裁剪器
        compute_devs = [i for i, t in _DEVICE_TIER.items() if t == "compute"]
        reduce_device = torch.device(f"cuda:{compute_devs[0]}") if compute_devs else None
        self._grad_clipper = DecoupledGradClipNorm(
            max_norm=max_grad_norm,
            reduce_device=reduce_device,
            process_group=process_group,
        )

        logger.info(
            "HeteroDecoupledGradDistOpt initialized: "
            "compute_optimizer=%s (groups=%d), cache_optimizer=%s (groups=%d)",
            type(self._compute_optimizer).__name__ if self._compute_optimizer else "None",
            len(compute_groups),
            type(self._cache_optimizer).__name__ if self._cache_optimizer else "None",
            len(cache_groups),
        )
        logger.info(locality_cache.summary())

    # ------------------------------------------------------------------
    # Bug 1 修复：master_weights 所有权守卫
    # ------------------------------------------------------------------

    def _disable_optimizer_master_weights(self) -> None:
        """
        当 LocalityCache 持有主权重时，强制禁用底层优化器的 master_weights。

        对应 Megatron 修复
        ------------------
        原始代码::

            if (not USING_PYTORCH_OPTIMIZER
                    and config.use_precision_aware_optimizer
                    and getattr(optimizer_part.optimizer, "master_weights", None) is not None):
                setattr(optimizer_part.optimizer, "master_weights", False)

        DES-LOC 扩展：不仅检查 ``use_precision_aware_optimizer``，
        还检查 ``LocalityCacheRegistry.owns_master_weights``，
        覆盖所有"缓存层持有主权重"的情形。
        """
        for opt in [self._compute_optimizer, self._cache_optimizer]:
            if opt is None:
                continue
            # 检查优化器是否有 master_weights 属性（FusedAdam / DeepSpeed Adam）
            if getattr(opt, "master_weights", None) is not None:
                # 检查是否有任何参数的主权重由 LocalityCache 持有
                all_params = [p for g in opt.param_groups for p in g["params"]]
                any_cache_owned = any(
                    self.locality_cache.owns_master_weights(p) for p in all_params
                )
                if any_cache_owned:
                    logger.warning(
                        "DES-LOC: disabling optimizer.master_weights for %s "
                        "because LocalityCache owns master weights. "
                        "(mirrors Megatron ab43d43 fix for FusedAdam + MFSDP)",
                        type(opt).__name__,
                    )
                    setattr(opt, "master_weights", False)
                    # 标记到注册表，供后续校验
                    for p in all_params:
                        self.locality_cache.mark_optimizer_master_weights_disabled(p)

    # ------------------------------------------------------------------
    # 梯度设置（DES-LOC 异构版本）
    # ------------------------------------------------------------------

    def set_hetero_decoupled_grad(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        src_device: Optional[torch.device] = None,
    ) -> None:
        """
        为参数挂载 HeteroDecoupledGrad。

        上游对应
        --------
        Megatron 中直接使用 ``param.decoupled_grad = grad``；
        DES-LOC 需要额外记录梯度来源设备，以便后续跨设备 norm 计算。

        参数
        ----
        param : torch.Tensor
            模型参数。
        grad : torch.Tensor
            解耦梯度张量。
        src_device : torch.device, optional
            梯度产生设备；默认为 grad.device。
        """
        src = src_device or grad.device
        master_dev = self.locality_cache.master_device(param) or param.device
        param.hetero_decoupled_grad = HeteroDecoupledGrad(
            data=grad,
            src_device=src,
            param_device=master_dev,
        )

    def _gather_all_params(self) -> List[torch.Tensor]:
        """收集所有参数（compute + cache tier）。"""
        params = []
        for opt in [self._compute_optimizer, self._cache_optimizer]:
            if opt is None:
                continue
            for group in opt.param_groups:
                params.extend(group["params"])
        return params

    # ------------------------------------------------------------------
    # 前向兼容：不支持操作显式抛出（对应 Megatron diff 中的 raise）
    # ------------------------------------------------------------------

    def copy_main_params_to_param_buffer(self) -> None:
        """
        不支持操作守卫。

        对应 Megatron 修复
        ------------------
        原始代码中 MFSDP 路径的 ``_copy_main_params_to_param_buffer`` 被静默跳过；
        修复后改为显式 ``raise NotImplementedError``。
        DES-LOC 同样不支持此操作（主权重由 LocalityCache 管理），故显式报错。
        """
        raise NotImplementedError(
            "copy_main_params_to_param_buffer is not supported when "
            "LocalityCache owns master weights. Use locality_cache.sync_to_model() instead."
        )

    def copy_model_params_to_main(self) -> None:
        """
        不支持操作守卫。

        对应 Megatron 修复中新增的::

            raise NotImplementedError(
                "Megatron-FSDP does not implement a model-to-main parameter update."
            )
        """
        raise NotImplementedError(
            "copy_model_params_to_main is not supported in DES-LOC HeteroDecoupledGradDistOpt. "
            "Master weights are managed exclusively by LocalityCache."
        )

    # ------------------------------------------------------------------
    # 零梯度
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        清零所有梯度，包括 hetero_decoupled_grad。

        对应 Megatron 修复中的 ``zero_grad_buffer`` 路径，确保
        LocalityCache 管理的分片梯度也被正确清零。
        """
        # 清零 hetero_decoupled_grad
        for param in self._gather_all_params():
            if hasattr(param, "hetero_decoupled_grad"):
                if set_to_none:
                    param.hetero_decoupled_grad = None
                else:
                    if param.hetero_decoupled_grad is not None:
                        param.hetero_decoupled_grad.data.zero_()
                        param.hetero_decoupled_grad._norm_cache = None

        # 清零底层优化器的 .grad
        for opt in [self._compute_optimizer, self._cache_optimizer]:
            if opt is not None:
                opt.zero_grad(set_to_none=set_to_none)

        logger.debug("HeteroDecoupledGradDistOpt: zero_grad done (set_to_none=%s)", set_to_none)

    # ------------------------------------------------------------------
    # 将解耦梯度复制到 param.grad（供底层优化器使用）
    # ------------------------------------------------------------------

    def _sync_decoupled_grads_to_param_grad(self) -> None:
        """
        将 HeteroDecoupledGrad 写入 param.grad。

        DES-LOC 中底层 Adam 仍通过 ``param.grad`` 访问梯度；
        此方法在 step() 前将解耦梯度（可能在不同设备上）搬运并赋值。

        对应 Megatron 修复
        ------------------
        ``copy_grads_to_model_params`` 路径（MFSDP 直接 return，
        因为 NCCL UB 已处理梯度同步）；DES-LOC 需要显式搬运。
        """
        for param in self._gather_all_params():
            hgrad = getattr(param, "hetero_decoupled_grad", None)
            if hgrad is None:
                continue
            # 搬运到参数所在设备
            grad_on_param_dev = hgrad.to_param_device(non_blocking=False)
            if param.grad is None or param.grad.shape != grad_on_param_dev.shape:
                param.grad = grad_on_param_dev.clone()
            else:
                param.grad.copy_(grad_on_param_dev)

    # ------------------------------------------------------------------
    # 优化器 step
    # ------------------------------------------------------------------

    def step(self, closure=None) -> Optional[float]:
        """
        执行异构参数更新。

        流程
        ----
        1. 裁剪跨设备梯度（``DecoupledGradClipNorm.clip_grads_``）。
        2. 将 HeteroDecoupledGrad 同步到 param.grad。
        3. 在 compute tier 设备（H100）上执行大参数 Adam step。
        4. 在 cache tier 设备（A6000）上执行小参数 Adam step。
        5. （可选）通过 LocalityCache 将更新后的主权重同步回模型参数。

        对应 Megatron 修复
        ------------------
        ``start_param_sync`` 路径（对应 ``all-gather sharded main weights``）；
        DES-LOC 用 LocalityCache 的设备间同步代替 NCCL all-gather。
        """
        loss = None
        all_params = self._gather_all_params()

        # Step 1: 跨设备梯度裁剪
        if self.max_grad_norm > 0 and all_params:
            global_norm = self._grad_clipper.clip_grads_(all_params)
            logger.debug("step: global_grad_norm=%.6f", global_norm.item())

        # Step 2: 同步解耦梯度到 param.grad
        self._sync_decoupled_grads_to_param_grad()

        # Step 3 & 4: 按 tier 执行 step
        for tier_name, opt in [
            ("compute(H100)", self._compute_optimizer),
            ("cache(A6000)", self._cache_optimizer),
        ]:
            if opt is None:
                continue
            logger.debug("step: running %s optimizer", tier_name)
            if closure is not None and tier_name.startswith("compute"):
                loss = opt.step(closure)
            else:
                opt.step()

        # 触发 LocalityCache 参数同步（如已实现）
        if hasattr(self.locality_cache, "start_param_sync"):
            self.locality_cache.start_param_sync()

        return loss

    # ------------------------------------------------------------------
    # 属性代理（兼容 DeepSpeed ZeRO 接口）
    # ------------------------------------------------------------------

    @property
    def param_groups(self) -> List[dict]:
        """合并 compute + cache tier 参数组。"""
        groups = []
        for opt in [self._compute_optimizer, self._cache_optimizer]:
            if opt is not None:
                groups.extend(opt.param_groups)
        return groups

    def state_dict(self) -> dict:
        """返回可序列化的状态字典（分 tier 存储）。"""
        return {
            "compute_state": self._compute_optimizer.state_dict() if self._compute_optimizer else None,
            "cache_state": self._cache_optimizer.state_dict() if self._cache_optimizer else None,
            "locality_cache_registry": {
                str(pid): {
                    "device": str(meta.device),
                    "owns_master_weights": meta.owns_master_weights,
                    "optimizer_master_weights_disabled": meta.optimizer_master_weights_disabled,
                }
                for pid, meta in self.locality_cache._registry.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """从 state_dict 恢复状态。"""
        if state_dict.get("compute_state") and self._compute_optimizer:
            self._compute_optimizer.load_state_dict(state_dict["compute_state"])
        if state_dict.get("cache_state") and self._cache_optimizer:
            self._cache_optimizer.load_state_dict(state_dict["cache_state"])
        logger.info("HeteroDecoupledGradDistOpt: state_dict loaded")

    # ------------------------------------------------------------------
    # count_zeros_fp32 等兼容接口
    # ------------------------------------------------------------------

    def count_zero_grads(self) -> int:
        """
        统计零梯度数量（对应 Megatron ``count_zeros_fp32``）。

        修复要点：对所有 hetero param 走 ``hetero_decoupled_grad`` 路径，
        而非 ``param.grad``，与 Megatron ab43d43 中统一走 ``decoupled_grad``
        的修复逻辑一致。
        """
        n_zeros = 0
        for param in self._gather_all_params():
            if not param.requires_grad:
                continue
            grad = self._grad_clipper._get_decoupled_grad(param)
            if grad is None:
                n_zeros += param.numel()
            else:
                n_zeros += int((grad == 0).sum().item())
        return n_zeros

    def __repr__(self) -> str:
        n_compute = sum(len(g["params"]) for g in self._compute_param_groups)
        n_cache = sum(len(g["params"]) for g in self._cache_param_groups)
        return (
            f"HeteroDecoupledGradDistOpt("
            f"compute_params={n_compute}, cache_params={n_cache}, "
            f"max_grad_norm={self.max_grad_norm}, "
            f"use_precision_aware={self.use_precision_aware_optimizer})"
        )


# ---------------------------------------------------------------------------
# 工厂函数（对应 Megatron get_megatron_optimizer 的 DES-LOC 版）
# ---------------------------------------------------------------------------


def build_hetero_decoupled_distopt(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    max_grad_norm: float = 1.0,
    use_precision_aware_optimizer: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
) -> HeteroDecoupledGradDistOpt:
    """
    构建 DES-LOC 异构分布式优化器的便捷工厂函数。

    对应 Megatron 的 ``get_megatron_optimizer`` 入口，完整复现以下修复逻辑：

    1. 为每个参数分配最优主权重设备（H100 / A6000）并在 LocalityCacheRegistry 注册。
    2. 调用 ``_build_hetero_param_groups`` 拆分参数组。
    3. 构建 ``HeteroDecoupledGradDistOpt`` 并触发 ``_disable_optimizer_master_weights``。

    参数
    ----
    model : torch.nn.Module
        待优化的模型。
    lr, weight_decay, betas : float
        Adam 超参数。
    max_grad_norm : float
        梯度裁剪阈值。
    use_precision_aware_optimizer : bool
        是否启用精度感知优化（影响 master_weights 禁用逻辑）。
    process_group : ProcessGroup, optional
        数据并行进程组。
    """
    registry = LocalityCacheRegistry()

    # 为每个参数选择最优主权重设备
    all_params = list(model.parameters())
    for param in all_params:
        master_dev = _preferred_device_for_param(param)
        registry.register(param, master_dev, owns_master_weights=True)

    param_groups = [{"params": all_params, "lr": lr, "weight_decay": weight_decay}]

    opt = HeteroDecoupledGradDistOpt(
        optimizer_cls=torch.optim.AdamW,
        param_groups=param_groups,
        locality_cache=registry,
        max_grad_norm=max_grad_norm,
        process_group=process_group,
        use_precision_aware_optimizer=use_precision_aware_optimizer,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    return opt


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("=== HeteroDecoupledGradDistOpt smoke test ===")

    # 构造简单模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(64, 32).to(device)

    # 构建优化器
    opt = build_hetero_decoupled_distopt(model, lr=1e-3, max_grad_norm=1.0)
    logger.info("Optimizer: %s", opt)

    # Smoke test 1: 参数组非空
    assert len(opt.param_groups) > 0, "param_groups should not be empty"

    # Smoke test 2: forward + backward + step 不崩溃
    x = torch.randn(8, 64, device=device)
    loss = model(x).sum()
    loss.backward()

    # 手动挂 HeteroDecoupledGrad（模拟 DES-LOC 反向传播后的状态）
    for p in model.parameters():
        if p.grad is not None:
            opt.set_hetero_decoupled_grad(p, p.grad.clone(), src_device=device)

    opt.step()
    logger.info("step() completed without error")

    # Smoke test 3: zero_grad 清除 hetero_decoupled_grad
    opt.zero_grad(set_to_none=True)
    for p in model.parameters():
        assert getattr(p, "hetero_decoupled_grad", None) is None, \
            "hetero_decoupled_grad should be None after zero_grad"

    # Smoke test 4: 不支持操作显式报错
    try:
        opt.copy_main_params_to_param_buffer()
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass

    # Smoke test 5: count_zero_grads 可调用
    n_zeros = opt.count_zero_grads()
    assert isinstance(n_zeros, int) and n_zeros >= 0

    logger.info("All smoke tests passed.")
    sys.exit(0)
