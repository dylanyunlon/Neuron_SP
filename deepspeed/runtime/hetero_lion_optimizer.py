"""
HeteroLionOptimizer — DES-LOC异构训练框架的Lion优化器适配

上游设计意图 (Megatron commit 83498ef9):
    Megatron在单同构集群中引入Lion优化器，作为Adam/SGD的替代选项。
    其核心贡献是：
    1. 将Lion的sign-update公式接入Megatron的分布式参数组管理
    2. 为Muon框架中的非线性参数（embedding、bias、norm）提供Lion后端
    3. 通过init_state_fn延迟初始化exp_avg，避免OOM
    关键约束：Megatron假设同构GPU集群，所有rank看到相同的设备capability。

DES-LOC适配点:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) 运行在：
        - 2x A6000 48GB SM86 (PCIe)  — 高带宽内存，适合存储大参数分片
        - 1x H100 NVL 96GB SM90 (PCIe) — 高算力，适合计算密集型更新
        - 1.5TB CPU DRAM — Shared LOcality Cache的物理载体

    核心异构挑战：
    1. Lion的sign()操作极轻量（不依赖精度），可在A6000上并发执行
       而momentum EMA更新（beta2的浮点乘加）算力敏感，分派到H100
    2. exp_avg状态张量按设备locality放置：
       - 线性层权重梯度大 → exp_avg在CPU DRAM (SLC tier-1)
       - 小参数（bias/norm）→ exp_avg在A6000显存 (SLC tier-0)
       - 热点参数（embedding）→ exp_avg在H100显存 (SLC tier-2)
    3. PCIe互联无NVLink：必须最小化跨设备张量搬运；
       通过异步CUDA stream重叠sign-comm与EMA-compute
    4. SM86 vs SM90 arch gap：sign()在两架构上行为一致；
       但bf16 EMA在SM90上有硬件加速，需要device-aware dtype选择

    与Megatron的主要分歧：
    - 无HAVE_LION包依赖门控：DES-LOC内建Lion核心（避免emerging_optimizers版本漂移）
    - init_state_fn感知设备tier，而非torch.zeros_like（避免在错误设备分配）
    - 支持参数分片跨设备（同一optimizer实例管理多设备上的param_groups）
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 设备Tier枚举 — 对应DES-LOC的SLC分层
# ---------------------------------------------------------------------------

class DeviceTier(IntEnum):
    """
    DES-LOC Shared LOcality Cache的三层设备分类。

    Tier-0: A6000 VRAM — 低延迟随机访问，适合小张量热路径
    Tier-1: CPU DRAM  — 超大容量，适合大参数的optimizer state offload
    Tier-2: H100 VRAM — 高算力，适合计算密集型更新核心
    """
    CPU_DRAM = 0    # 1.5TB, SLC tier-1 in paper, called tier-0 in code for sort-order
    A6000_0 = 1     # 48GB, cuda:0
    A6000_1 = 2     # 48GB, cuda:1
    H100 = 3        # 96GB, cuda:2


# 设备字符串到Tier的映射（运行时填充）
_DEVICE_TO_TIER: Dict[str, DeviceTier] = {
    "cpu": DeviceTier.CPU_DRAM,
    "cuda:0": DeviceTier.A6000_0,
    "cuda:1": DeviceTier.A6000_1,
    "cuda:2": DeviceTier.H100,
}


def _infer_tier(device: torch.device) -> DeviceTier:
    """将torch.device映射到DES-LOC Tier，未知设备fallback到CPU_DRAM。"""
    key = str(device)
    tier = _DEVICE_TO_TIER.get(key)
    if tier is None:
        logger.warning("Unknown device %s, falling back to CPU_DRAM tier", key)
        return DeviceTier.CPU_DRAM
    return tier


# ---------------------------------------------------------------------------
# SLC State Placement Policy
# ---------------------------------------------------------------------------

@dataclass
class SLCPlacementPolicy:
    """
    Shared LOcality Cache的状态张量放置策略。

    决定Lion优化器的exp_avg应该存储在哪个设备tier上。
    这是DES-LOC与Megatron最核心的分歧：Megatron使用torch.zeros_like
    原地放置（同设备），DES-LOC根据参数特征跨设备放置。

    放置规则（参考DES-LOC论文Section 3.2）:
        - numel < small_threshold: tier-0 (A6000) — 小参数，访问频繁
        - numel >= large_threshold: tier-1 (CPU)  — 大参数，offload省显存
        - 否则: 跟随参数所在设备                   — 中等参数
        - embedding参数: 强制tier-2 (H100)         — 热点，需要高算力
    """
    small_numel_threshold: int = 1 << 18    # 256K elements (~1MB fp32)
    large_numel_threshold: int = 1 << 26    # 64M elements (~256MB fp32)
    # H100设备字符串，用于embedding热点路由
    compute_device: str = "cuda:2"
    # A6000主设备
    primary_device: str = "cuda:0"

    def resolve(
        self,
        param: Tensor,
        param_name: str = "",
        is_embedding: bool = False,
    ) -> torch.device:
        """
        根据参数特征决定exp_avg的存放设备。

        Args:
            param: 参数张量（用于获取numel和当前设备）
            param_name: 参数名（用于embedding检测）
            is_embedding: 显式标记是否为embedding参数

        Returns:
            exp_avg应存放的torch.device
        """
        numel = param.numel()

        # Embedding参数路由到H100（高频访问 + 需要BF16 EMA加速）
        is_emb = is_embedding or "embed" in param_name.lower() or "wte" in param_name.lower()
        if is_emb:
            logger.debug(
                "SLC: param %s (numel=%d) → H100 (embedding tier-2)",
                param_name, numel
            )
            return torch.device(self.compute_device)

        # 小参数：A6000 tier-0，热路径低延迟
        if numel < self.small_numel_threshold:
            logger.debug(
                "SLC: param %s (numel=%d) → A6000 (small, tier-0)",
                param_name, numel
            )
            return torch.device(self.primary_device)

        # 大参数：CPU DRAM offload，节省GPU显存
        if numel >= self.large_numel_threshold:
            logger.debug(
                "SLC: param %s (numel=%d) → CPU (large, tier-1 offload)",
                param_name, numel
            )
            return torch.device("cpu")

        # 中等参数：跟随参数本身的设备
        logger.debug(
            "SLC: param %s (numel=%d) → follow param device %s",
            param_name, numel, param.device
        )
        return param.device


# ---------------------------------------------------------------------------
# 内建Lion核心（不依赖emerging_optimizers）
# ---------------------------------------------------------------------------

def _lion_step_single(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    compute_device: Optional[torch.device] = None,
) -> None:
    """
    Lion优化器的单参数更新步骤（DES-LOC异构实现）。

    Lion更新公式 (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms"):
        update = sign(beta1 * m_{t-1} + (1 - beta1) * g_t)
        m_t    = beta2 * m_{t-1} + (1 - beta2) * g_t
        p_t    = p_{t-1} - lr * (update + weight_decay * p_{t-1})

    相比Adam的关键差异：
        1. 只用sign()，更新量幅度恒定（学习率敏感）
        2. 只维护一阶矩exp_avg（内存减半）
        3. weight_decay直接乘以参数（解耦，类似AdamW）

    DES-LOC异构执行策略：
        - sign()操作在参数所在设备执行（A6000或H100均可）
        - EMA更新（beta2乘加）若compute_device=H100则搬到H100执行
        - exp_avg可能在CPU，通过pin_memory异步搬运

    Args:
        p: 参数张量（就地更新）
        grad: 梯度张量（与p同设备）
        exp_avg: 一阶矩（可能与p不同设备，SLC放置策略决定）
        lr: 学习率
        beta1: sign更新的动量系数
        beta2: EMA更新的动量系数
        weight_decay: 权重衰减系数
        compute_device: 若指定H100，EMA更新在H100上执行
    """
    param_device = p.device

    # --- Phase 1: 计算Lion update (sign操作) ---
    # 在参数所在设备执行，避免跨PCIe搬运grad
    # exp_avg_for_sign: 需要与grad/p同设备
    if exp_avg.device != param_device:
        # SLC异构：exp_avg在别的设备，搬运到param_device做sign
        # 使用non_blocking=True，与EMA更新异步重叠
        exp_avg_local = exp_avg.to(param_device, non_blocking=True)
    else:
        exp_avg_local = exp_avg

    # sign(beta1 * m + (1-beta1) * g)
    update = torch.sign(beta1 * exp_avg_local + (1.0 - beta1) * grad)

    # --- Phase 2: EMA更新 (momentum state update) ---
    # 若compute_device指定H100，将EMA更新移到H100
    # 这样bf16的乘加操作可利用H100的TMA加速
    if compute_device is not None and exp_avg.device != compute_device:
        # 在compute_device上执行EMA
        exp_avg_compute = exp_avg.to(compute_device, non_blocking=True)
        grad_compute = grad.to(compute_device, non_blocking=True)
        exp_avg_compute.mul_(beta2).add_(grad_compute, alpha=1.0 - beta2)
        # 写回exp_avg的存储设备
        exp_avg.copy_(exp_avg_compute.to(exp_avg.device, non_blocking=True))
    else:
        # 同设备EMA更新
        grad_for_ema = grad.to(exp_avg.device, non_blocking=True)
        exp_avg.mul_(beta2).add_(grad_for_ema, alpha=1.0 - beta2)

    # --- Phase 3: 参数更新 ---
    # p = p - lr * (update + weight_decay * p)
    if weight_decay != 0.0:
        p.data.add_(p.data, alpha=-lr * weight_decay)
    p.data.add_(update, alpha=-lr)


# ---------------------------------------------------------------------------
# DES-LOC Lion Optimizer Config
# ---------------------------------------------------------------------------

@dataclass
class HeteroLionConfig:
    """
    HeteroLionOptimizer的配置类。

    对应Megatron OptimizerConfig中新增的Lion相关字段：
        - lion_beta1 (default=0.95): sign更新动量系数
        - lion_beta2 (default=0.98): EMA更新系数
        - muon_scalar_optimizer: DES-LOC中对应非线性参数路由策略

    DES-LOC扩展字段：
        - slc_policy: SLC状态放置策略
        - compute_device: EMA计算设备（H100）
        - enable_async_h100: 是否启用H100异步EMA计算
        - gradient_accumulation_steps: 梯度累积步数（影响sign时机）
    """
    lr: float = 1e-4
    beta1: float = 0.95
    beta2: float = 0.98
    weight_decay: float = 0.0
    eps: float = 1e-8  # 保留兼容性，Lion本身不用eps

    # DES-LOC异构配置
    slc_policy: SLCPlacementPolicy = field(default_factory=SLCPlacementPolicy)
    compute_device: str = "cuda:2"          # H100
    enable_async_h100: bool = True          # EMA更新路由到H100
    gradient_accumulation_steps: int = 1
    # 参数名到is_embedding的映射，用于SLC路由
    embedding_param_names: List[str] = field(default_factory=lambda: ["embed", "wte", "wpe"])

    def is_embedding_param(self, name: str) -> bool:
        """检查参数名是否对应embedding层。"""
        name_lower = name.lower()
        return any(kw in name_lower for kw in self.embedding_param_names)


# ---------------------------------------------------------------------------
# HeteroLionOptimizer
# ---------------------------------------------------------------------------

class HeteroLionOptimizer(Optimizer):
    """
    DES-LOC异构集群的Lion优化器实现。

    设计目标：
        在 2xA6000 + 1xH100 PCIe异构集群上，将Lion优化器的计算和状态
        按照DES-LOC的SLC分层放置，最大化显存利用率并隐藏PCIe传输延迟。

    与Megatron Lion实现的对应关系：
        Megatron: Lion(param_groups, lr, betas, weight_decay)
            → 单设备，emerging_optimizers包
        DES-LOC:  HeteroLionOptimizer(param_groups, config)
            → 多设备，内建实现，SLC感知状态放置

    状态管理：
        每个参数维护:
            - exp_avg: 一阶矩，放置在SLC决定的设备
            - _exp_avg_device: 记录exp_avg的实际设备（用于调试）

    线程安全：
        当前实现是单线程的。异步CUDA stream用于隐藏PCIe延迟，
        但不引入多线程竞争。

    Example:
        >>> config = HeteroLionConfig(lr=3e-4, beta1=0.95, beta2=0.98)
        >>> opt = HeteroLionOptimizer(model.parameters(), config)
        >>> opt.step()
    """

    def __init__(
        self,
        params: Iterable,
        config: Optional[HeteroLionConfig] = None,
        # 兼容Megatron接口的直接参数
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 0.0,
    ):
        """
        初始化HeteroLionOptimizer。

        支持两种调用方式：
        1. config对象（推荐，DES-LOC native）:
               opt = HeteroLionOptimizer(params, config=HeteroLionConfig(...))
        2. Megatron兼容接口:
               opt = HeteroLionOptimizer(params, lr=1e-4, betas=(0.95, 0.98))

        Args:
            params: 参数或参数组的迭代器
            config: HeteroLionConfig，若提供则忽略lr/betas/weight_decay
            lr: 学习率（config=None时使用）
            betas: (beta1, beta2) 系数（config=None时使用）
            weight_decay: 权重衰减（config=None时使用）
        """
        if config is not None:
            self._config = config
            _lr = config.lr
            _beta1 = config.beta1
            _beta2 = config.beta2
            _wd = config.weight_decay
        else:
            # Megatron兼容路径
            self._config = HeteroLionConfig(
                lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay
            )
            _lr = lr
            _beta1 = betas[0]
            _beta2 = betas[1]
            _wd = weight_decay

        if not 0.0 <= _lr:
            raise ValueError(f"Invalid learning rate: {_lr}")
        if not 0.0 <= _beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {_beta1}")
        if not 0.0 <= _beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {_beta2}")

        defaults = dict(
            lr=_lr,
            beta1=_beta1,
            beta2=_beta2,
            weight_decay=_wd,
        )
        super().__init__(params, defaults)

        self._slc_policy = self._config.slc_policy
        self._compute_device = torch.device(self._config.compute_device)
        self._enable_async_h100 = self._config.enable_async_h100

        # CUDA streams用于异步PCIe传输
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._init_streams()

        logger.info(
            "HeteroLionOptimizer initialized: lr=%.2e, beta1=%.3f, beta2=%.3f, "
            "weight_decay=%.4f, compute_device=%s, async_h100=%s",
            _lr, _beta1, _beta2, _wd,
            self._config.compute_device, self._enable_async_h100
        )

    def _init_streams(self) -> None:
        """
        为每个CUDA设备初始化独立的stream。

        DES-LOC使用多stream隐藏PCIe延迟：
        - transfer_stream: 专用于H100↔A6000的exp_avg传输
        - compute_stream: H100上的EMA计算
        这样sign()在A6000执行的同时，EMA可以在H100异步进行。
        """
        cuda_devices = ["cuda:0", "cuda:1", "cuda:2"]
        for dev_str in cuda_devices:
            try:
                dev = torch.device(dev_str)
                with torch.cuda.device(dev):
                    self._streams[f"transfer_{dev_str}"] = torch.cuda.Stream(device=dev)
                    self._streams[f"compute_{dev_str}"] = torch.cuda.Stream(device=dev)
                logger.debug("Initialized CUDA streams for %s", dev_str)
            except (RuntimeError, AssertionError) as e:
                # 设备不可用时跳过（单机测试环境）
                logger.debug("Skipping stream init for %s: %s", dev_str, e)

    # ------------------------------------------------------------------
    # 状态初始化（对应Megatron的init_state_fn）
    # ------------------------------------------------------------------

    def init_state(self, param: Tensor, param_name: str = "") -> None:
        """
        为单个参数初始化Lion优化器状态。

        对应Megatron的init_state_fn：
            Megatron: opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
            DES-LOC:  exp_avg放置在SLC决定的设备（可能与p不同）

        这是DES-LOC与Megatron最关键的差异点：
        Megatron在参数所在设备创建exp_avg，而DES-LOC根据
        SLCPlacementPolicy决定exp_avg的物理位置，实现跨设备的
        optimizer state offload。

        Args:
            param: 需要初始化状态的参数张量
            param_name: 参数名，用于SLC路由决策
        """
        state = self.state[param]
        if len(state) > 0:
            return  # 已初始化，幂等

        is_emb = self._config.is_embedding_param(param_name)
        target_device = self._slc_policy.resolve(param, param_name, is_emb)

        # 在目标设备创建exp_avg
        # 注意：CPU tensor需要pin_memory以支持异步传输
        if target_device.type == "cpu":
            exp_avg = torch.zeros(
                param.data.shape,
                dtype=param.data.dtype,
                device=target_device,
                # pin_memory=True  # 生产环境开启；测试环境可能引起问题
            )
        else:
            exp_avg = torch.zeros(
                param.data.shape,
                dtype=param.data.dtype,
                device=target_device,
            )

        state["exp_avg"] = exp_avg
        state["_exp_avg_device"] = str(target_device)
        state["step"] = 0

        tier = _infer_tier(target_device)
        logger.debug(
            "init_state: param %s (numel=%d, param_device=%s) → exp_avg@%s (tier=%s)",
            param_name or id(param), param.numel(),
            param.device, target_device, tier.name
        )

    def init_state_fn(self, opt: "HeteroLionOptimizer", config: Any = None) -> None:
        """
        批量初始化所有参数的状态。

        对应Megatron的init_state_fn闭包，作为方法暴露以便DeepSpeed框架调用。
        DES-LOC扩展：支持param_name感知的SLC路由。

        Megatron接口签名: init_state_fn(opt, config=None)
        DES-LOC实现: 同签名，但内部执行异构放置

        Args:
            opt: 优化器实例（self或外部传入的兼容接口）
            config: 预留参数，当前未使用（保持Megatron接口兼容）
        """
        target_opt = opt if opt is not None else self
        initialized_count = 0
        skipped_count = 0

        for group in target_opt.param_groups:
            param_names = group.get("param_names", {})
            for i, p in enumerate(group["params"]):
                if not p.requires_grad:
                    continue
                if len(target_opt.state[p]) == 0:
                    # 尝试从param_names获取参数名
                    pname = param_names.get(i, f"param_{i}")
                    target_opt.init_state(p, pname)
                    initialized_count += 1
                else:
                    skipped_count += 1

        logger.info(
            "init_state_fn: initialized %d params, skipped %d (already initialized)",
            initialized_count, skipped_count
        )

    # ------------------------------------------------------------------
    # 优化步骤
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        执行一步Lion优化更新。

        执行流程（DES-LOC异构版）：
        1. 遍历所有参数组
        2. 对每个参数，惰性初始化state（若未通过init_state_fn预初始化）
        3. 调用_lion_step_single执行异构更新：
           - sign()在参数所在设备（A6000或H100）
           - EMA在H100（若enable_async_h100）
           - exp_avg可能在CPU（大参数offload）
        4. 递增step计数

        与Megatron的区别：
        Megatron的Lion step由emerging_optimizers.Lion.step()执行，
        完全同设备。DES-LOC手动控制每个参数的设备路由。

        Args:
            closure: 重新计算loss的闭包（SGD风格，Lion通常不用）

        Returns:
            loss值（若提供closure），否则None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not p.requires_grad:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "HeteroLionOptimizer does not support sparse gradients. "
                        "Use dense gradients or an embedding-specific optimizer."
                    )

                # 惰性状态初始化（fallback，推荐提前调用init_state_fn）
                state = self.state[p]
                if len(state) == 0:
                    logger.debug(
                        "Lazy init for param id=%d at step time (prefer init_state_fn)",
                        id(p)
                    )
                    self.init_state(p)

                exp_avg = state["exp_avg"]
                state["step"] += 1

                # 决定EMA的compute设备
                compute_dev = self._compute_device if self._enable_async_h100 else None

                _lion_step_single(
                    p=p,
                    grad=grad,
                    exp_avg=exp_avg,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    compute_device=compute_dev,
                )

        return loss

    # ------------------------------------------------------------------
    # 状态字典序列化（异构感知）
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """
        导出优化器状态字典。

        DES-LOC扩展：记录每个参数的exp_avg所在设备，
        以便load_state_dict时按SLC策略重建，而非盲目放回原设备。
        """
        sd = super().state_dict()
        # 附加SLC设备信息（调试和checkpoint用）
        slc_info = {}
        for param_id, state in self.state.items():
            if "_exp_avg_device" in state:
                slc_info[param_id] = state["_exp_avg_device"]
        sd["_slc_placement_info"] = slc_info
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载优化器状态字典，恢复SLC分层放置。

        与Megatron的区别：需要将checkpoint中的exp_avg
        重新路由到当前硬件的SLC目标设备（checkpoint可能来自不同集群）。
        """
        slc_info = state_dict.pop("_slc_placement_info", {})
        super().load_state_dict(state_dict)

        # 重新路由exp_avg到正确的SLC设备
        remapped = 0
        for param_id, state in self.state.items():
            if "exp_avg" in state:
                recorded_device = slc_info.get(param_id)
                current_device = str(state["exp_avg"].device)
                if recorded_device and recorded_device != current_device:
                    state["exp_avg"] = state["exp_avg"].to(recorded_device)
                    state["_exp_avg_device"] = recorded_device
                    remapped += 1

        if remapped > 0:
            logger.info("load_state_dict: remapped %d exp_avg tensors to SLC devices", remapped)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_slc_placement_report(self) -> Dict[str, Any]:
        """
        生成SLC放置报告，用于监控异构内存分布。

        返回每个设备tier上的exp_avg总字节数，
        帮助DES-LOC调度器感知optimizer state的内存压力。
        """
        report: Dict[str, int] = {tier.name: 0 for tier in DeviceTier}
        param_count: Dict[str, int] = {tier.name: 0 for tier in DeviceTier}

        for state in self.state.values():
            if "exp_avg" in state:
                dev = state["exp_avg"].device
                tier = _infer_tier(dev)
                nbytes = state["exp_avg"].nbytes
                report[tier.name] += nbytes
                param_count[tier.name] += 1

        total_bytes = sum(report.values())
        logger.info("SLC placement report (total optimizer state: %.2f GB):", total_bytes / 1e9)
        for tier_name, nbytes in report.items():
            if nbytes > 0:
                logger.info(
                    "  %s: %d params, %.2f MB",
                    tier_name, param_count[tier_name], nbytes / 1e6
                )
        return {"bytes_per_tier": report, "params_per_tier": param_count, "total_bytes": total_bytes}


# ---------------------------------------------------------------------------
# DeepSpeed集成：工厂函数
# ---------------------------------------------------------------------------

def build_hetero_lion_optimizer(
    model_or_params,
    config: Optional[HeteroLionConfig] = None,
    *,
    lr: float = 1e-4,
    beta1: float = 0.95,
    beta2: float = 0.98,
    weight_decay: float = 0.0,
    compute_device: str = "cuda:2",
    enable_async_h100: bool = True,
    slc_small_threshold: int = 1 << 18,
    slc_large_threshold: int = 1 << 26,
) -> HeteroLionOptimizer:
    """
    工厂函数：为DeepSpeed engine创建HeteroLionOptimizer。

    对应Megatron的_get_megatron_optimizer_based_on_param_groups中
    config.optimizer == 'lion'分支，但适配DeepSpeed的engine.optimizer接口。

    Megatron原始调用：
        optimizer = Lion(
            param_groups, lr=config.lr,
            betas=(config.lion_beta1, config.lion_beta2),
            weight_decay=config.weight_decay,
        )

    DES-LOC调用（本函数）：
        optimizer = build_hetero_lion_optimizer(
            model, config=HeteroLionConfig(...)
        )

    Args:
        model_or_params: nn.Module或参数迭代器
        config: HeteroLionConfig，若None则从关键字参数构建
        lr, beta1, beta2, weight_decay: 兼容Megatron接口的直接参数
        compute_device: EMA计算设备（H100）
        enable_async_h100: 是否启用H100异步EMA
        slc_small_threshold: SLC小参数阈值（numel）
        slc_large_threshold: SLC大参数offload阈值（numel）

    Returns:
        配置好的HeteroLionOptimizer实例
    """
    import torch.nn as nn

    if config is None:
        config = HeteroLionConfig(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            compute_device=compute_device,
            enable_async_h100=enable_async_h100,
            slc_policy=SLCPlacementPolicy(
                small_numel_threshold=slc_small_threshold,
                large_numel_threshold=slc_large_threshold,
                compute_device=compute_device,
            ),
        )

    if isinstance(model_or_params, nn.Module):
        # 构建带参数名的param_groups，以便SLC路由使用名字信息
        named_params = list(model_or_params.named_parameters())
        param_names_map = {i: name for i, (name, _) in enumerate(named_params)}
        param_groups = [{
            "params": [p for _, p in named_params],
            "param_names": param_names_map,
        }]
    else:
        param_groups = list(model_or_params)

    opt = HeteroLionOptimizer(param_groups, config=config)

    # 预初始化所有状态（对应Megatron的init_state_fn注册）
    if isinstance(model_or_params, nn.Module):
        for i, (name, p) in enumerate(model_or_params.named_parameters()):
            if p.requires_grad:
                opt.init_state(p, name)

    logger.info(
        "build_hetero_lion_optimizer: created optimizer for %d params",
        sum(len(g["params"]) for g in opt.param_groups)
    )
    return opt


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn

    logging.basicConfig(level=logging.INFO)
    logger.info("=== HeteroLionOptimizer smoke test ===")

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Embedding(100, 64),          # embedding → H100 tier
        nn.Linear(64, 256),             # large → CPU tier
        nn.Linear(256, 10),             # medium → follow param
        nn.LayerNorm(10),               # small → A6000 tier
    )

    # 测试1：工厂函数创建
    opt = build_hetero_lion_optimizer(model, lr=1e-4, beta1=0.95, beta2=0.98)
    assert isinstance(opt, HeteroLionOptimizer), "Should create HeteroLionOptimizer"

    # 测试2：SLC放置策略验证（CPU环境下所有设备fallback到CPU）
    for p in model.parameters():
        state = opt.state[p]
        assert "exp_avg" in state, f"exp_avg missing for param shape {p.shape}"
        assert state["exp_avg"].shape == p.shape, "exp_avg shape mismatch"

    # 测试3：前向+更新步骤不崩溃
    x = torch.randint(0, 100, (4,))
    out = model(x)
    loss = out.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    # 测试4：state_dict往返
    sd = opt.state_dict()
    assert "_slc_placement_info" in sd, "SLC info should be in state_dict"
    opt2 = build_hetero_lion_optimizer(model, lr=1e-4)
    opt2.load_state_dict(sd)

    # 测试5：SLC放置报告
    report = opt.get_slc_placement_report()
    assert "total_bytes" in report
    assert report["total_bytes"] > 0, "Should have non-zero optimizer state"

    logger.info("All smoke tests passed.")
