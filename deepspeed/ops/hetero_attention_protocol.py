"""
DES-LOC Heterogeneous Attention Protocol Layer
===============================================

上游设计意图 (Megatron commit 90e685b):
    Megatron 原始 commit 将 SelfAttention/CrossAttention 的子模块注册方式从
    ModuleSpec（一个描述性的数据类，依赖 build_module() 在运行时反射构造）替换为
    结构化子类型协议（typing.Protocol）。核心动机：
      1. 让 linear_qkv / core_attention 等子模块的构造签名在静态类型检查阶段可见，
         消除隐式 Any；
      2. 引入 not_none() 断言守卫，在导入时（而非前向传播时）就捕获 TE 缺失的错误；
      3. 把 build_module(submodules.core_attention, ...) 改写为
         submodules.core_attention(...)，让调用链对 mypy/pyright 完全透明。

DES-LOC 适配点:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) 的核心矛盾在于：
    同一个训练 step 中，attention 计算可能分散在三块算力差异悬殊的设备上：
      - A6000 x2 (SM86, 48 GB, PCIe)  —— 承载 embedding + FFN 的大部分
      - H100 NVL  (SM90, 96 GB, PCIe) —— 承载 core attention（FlashAttention-3）

    Megatron 的 Protocol 重构恰好给了我们一个干净的注入点：
      * LinearQkvBuilder / CoreAttentionBuilder 这两个 Protocol 成为
        "设备路由元数据"的载体——构造时决定算子落在哪个设备；
      * HeteroAttentionProtocol 在 __init__ 中根据 DevicePlacementPolicy
        决定把 linear_qkv 建在 A6000 上、把 core_attention 建在 H100 上；
      * SharedLocalityCache (SLC) 在两块设备之间扮演 L3-like staging buffer，
        缓存 KV projection 输出，避免跨 PCIe 的冗余传输；
      * backward_dw() 协议方法对应 delay_wgrad_compute 路径：
        weight gradient 的 all-reduce 被解耦到 forward 完成之后，
        使得 H100 的 core_attn backward 可以和 A6000 的 linear_qkv backward 流水。

文件结构:
    1. Protocol 定义 (LinearQkvBuilder, CoreAttentionBuilder, ...)
    2. DevicePlacementPolicy — 决定算子落在哪个 CUDA 设备
    3. SharedLocalityCache (SLC) — 跨设备 PCIe staging buffer
    4. HeteroAttentionProtocol — DES-LOC 异构 self-attention 主体
    5. HeteroCrossAttentionProtocol — DES-LOC 异构 cross-attention 主体
    6. 工厂函数 build_hetero_attention_spec
    7. Smoke test
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# ---------------------------------------------------------------------------
# not_none — mirrors Megatron typed_torch.not_none
# ---------------------------------------------------------------------------
def not_none(value: Optional[T]) -> T:
    """Assert that *value* is not None and return it.

    上游对应: megatron/core/typed_torch.py :: not_none()
    DES-LOC 用途: 在构造阶段（而非 forward）提前暴露 TE / backend 缺失，
    避免在异构设备切换时 fallback 到 None 引发难以定位的运行时错误。
    """
    if value is None:
        raise ValueError(
            "[DES-LOC] not_none() received None. "
            "Check that the required backend (TE / Kitchen / baseline) is installed "
            "and that the correct DevicePlacementPolicy was chosen."
        )
    return value


# ---------------------------------------------------------------------------
# Attention mask type (minimal enum mirroring Megatron)
# ---------------------------------------------------------------------------
from enum import Enum


class AttnMaskType(Enum):
    """Mirrors megatron.core.transformer.enums.AttnMaskType."""
    padding = "padding"
    causal  = "causal"
    no_mask = "no_mask"


# ---------------------------------------------------------------------------
# Protocols  (mirrors Megatron attention.py Protocol section)
# ---------------------------------------------------------------------------

@runtime_checkable
class LinearQkv(Protocol):
    """Protocol for a linear_qkv module on the *projection* device.

    上游: megatron/core/transformer/attention.py :: LinearQkv
    DES-LOC: 该 protocol 的实现必须运行在 DevicePlacementPolicy.proj_device 上
    (通常是 A6000)，输出 tensor 通过 SharedLocalityCache 异步搬运到 H100。
    """

    def forward(self, input: torch.Tensor, /) -> Tuple[torch.Tensor, object]:
        """Returns (output, bias_or_None)."""
        ...

    def backward_dw(self) -> None:
        """Delay weight gradient — called after forward completes on H100.

        DES-LOC: 解耦 weight grad 计算，使 H100 core_attn backward 与
        A6000 linear_qkv wgrad all-reduce 可以并发执行。
        """
        ...


@runtime_checkable
class LinearQkvBuilder(Protocol):
    """Factory protocol for constructing LinearQkv layers.

    上游: megatron/core/transformer/attention.py :: LinearQkvBuilder
    DES-LOC: 工厂函数在 HeteroAttentionProtocol.__init__ 中被调用时，
    已知目标 device，因此实现者可以把 .to(device) 包装进 __call__。
    """

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: "HeteroTransformerConfig",
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        device: Optional[torch.device] = None,   # DES-LOC 扩展字段
    ) -> LinearQkv:
        ...


@runtime_checkable
class CoreAttention(Protocol):
    """Protocol for core attention computation on the *attention* device.

    上游: megatron/core/transformer/attention.py :: CoreAttention
    DES-LOC: 该 protocol 的实现运行在 DevicePlacementPolicy.attn_device 上
    (H100 NVL)，接收从 SLC 取出的 Q/K/V tensor（已在 H100 显存中）。
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        /,
        *,
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor],
        packed_seq_params: Optional[object],
    ) -> torch.Tensor:
        ...


@runtime_checkable
class CoreAttentionBuilder(Protocol):
    """Factory protocol for constructing CoreAttention layers.

    上游: megatron/core/transformer/attention.py :: CoreAttentionBuilder
    DES-LOC: 工厂内部可根据 device.type + compute_capability 自动选择
    FlashAttention-3 (H100 SM90) 或 FlashAttention-2 (A6000 SM86) backend。
    """

    def __call__(
        self,
        *,
        config: "HeteroTransformerConfig",
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str],
        softmax_scale: Optional[float],
        pg_collection: Optional[object],
        device: Optional[torch.device] = None,   # DES-LOC 扩展字段
    ) -> CoreAttention:
        ...


# ---------------------------------------------------------------------------
# Minimal config stub  (replace with real DeepSpeed config in production)
# ---------------------------------------------------------------------------

@dataclass
class HeteroTransformerConfig:
    """Minimal transformer config consumed by DES-LOC attention modules.

    In production this should inherit / wrap DeepSpeed's TransformerConfig.
    Fields mirror the subset used in Megatron's Attention.__init__.
    """
    hidden_size: int           = 4096
    num_attention_heads: int   = 32
    num_query_groups: int      = 32    # GQA: set < num_attention_heads for MQA/GQA
    kv_channels: int           = 128
    add_bias_linear: bool      = False
    add_qkv_bias: bool         = False
    attention_output_gate: bool = False
    init_method: Optional[Callable[[torch.Tensor], None]] = None
    batch_invariant_mode: bool  = False
    # DES-LOC specific
    slc_capacity_mb: float      = 512.0   # SharedLocalityCache DRAM budget per layer (MB)
    delay_wgrad_compute: bool   = True    # decouple weight grad from activation grad


# ---------------------------------------------------------------------------
# DevicePlacementPolicy
# ---------------------------------------------------------------------------

@dataclass
class DevicePlacementPolicy:
    """Maps attention sub-operations to physical devices.

    DES-LOC 设计:
      proj_device  — linear_qkv / linear_proj 运行设备（A6000，显存充裕，适合大矩阵乘）
      attn_device  — core_attention 运行设备（H100，SM90 FlashAttn-3 加速）
      slc_device   — SharedLocalityCache 所在设备（CPU DRAM，容量 1.5 TB，兜底缓冲）

    PCIe 互联（无 NVLink）意味着跨设备搬运是瓶颈，SLC 通过固定住 KV 输出的生命周期
    来减少冗余传输次数：同一 layer 的 forward 和 backward 共享同一份 KV 副本。
    """
    proj_device: torch.device  = field(default_factory=lambda: torch.device("cuda:0"))
    attn_device: torch.device  = field(default_factory=lambda: torch.device("cuda:2"))
    slc_device:  torch.device  = field(default_factory=lambda: torch.device("cpu"))

    @classmethod
    def auto_detect(cls) -> "DevicePlacementPolicy":
        """Heuristic: assign H100 (SM90) to attn_device, A6000s to proj_device.

        Requires at least 1 CUDA device. Falls back to cpu if GPU count < 2.
        """
        if not torch.cuda.is_available():
            logger.warning("[DES-LOC] No CUDA devices found; using CPU for all placements.")
            cpu = torch.device("cpu")
            return cls(proj_device=cpu, attn_device=cpu, slc_device=cpu)

        n = torch.cuda.device_count()
        sm_caps: List[Tuple[int, int, int]] = []   # (major, minor, idx)
        for i in range(n):
            cap = torch.cuda.get_device_capability(i)
            sm_caps.append((cap[0], cap[1], i))
            logger.debug(
                "[DES-LOC] GPU %d: %s  SM%d%d",
                i, torch.cuda.get_device_name(i), cap[0], cap[1],
            )

        # H100 NVL → SM90; prefer it for core attention (FlashAttention-3)
        h100_candidates = [idx for (maj, min_, idx) in sm_caps if maj >= 9]
        a6000_candidates = [idx for (maj, min_, idx) in sm_caps if maj < 9]

        attn_idx  = h100_candidates[0]  if h100_candidates  else sm_caps[-1][2]
        proj_idx  = a6000_candidates[0] if a6000_candidates else sm_caps[0][2]

        logger.info(
            "[DES-LOC] DevicePlacementPolicy: proj=cuda:%d  attn=cuda:%d  slc=cpu",
            proj_idx, attn_idx,
        )
        return cls(
            proj_device=torch.device(f"cuda:{proj_idx}"),
            attn_device=torch.device(f"cuda:{attn_idx}"),
            slc_device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# SharedLocalityCache (SLC)
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """PCIe-aware staging buffer between projection device and attention device.

    DES-LOC 核心组件:
      SLC 作为一个逻辑上的 L3 缓存，驻留在 CPU DRAM（1.5 TB），
      在以下场景中减少 PCIe 往返：

      1. Forward pass:
         A6000 完成 linear_qkv → QKV tensor pin_memory 到 CPU SLC
         → H100 异步 prefetch（non_blocking=True）
         这样 H100 在 A6000 算完之前就可以开始 DMA 传输。

      2. Backward pass:
         SLC 保留 KV 副本，H100 backward 不必重新从 A6000 拉取。
         对应 Megatron delay_wgrad_compute 路径，backward_dw() 解耦
         weight gradient 到 SLC flush 之后再触发。

      3. Capacity management:
         超出 slc_capacity_mb 时，LRU 逐出最老的条目（layer_number 最小）。

    线程安全: 使用 threading.Lock 保护 _store 字典。
    """

    def __init__(self, capacity_mb: float = 512.0, device: torch.device = torch.device("cpu")):
        self.capacity_bytes = int(capacity_mb * 1024 * 1024)
        self.device = device
        self._store: dict[str, torch.Tensor] = {}
        self._order: List[str] = []          # LRU order (oldest first)
        self._used_bytes: int = 0
        self._lock = threading.Lock()
        logger.info(
            "[DES-LOC][SLC] Initialized  capacity=%.1f MB  device=%s",
            capacity_mb, device,
        )

    # ------------------------------------------------------------------
    def _tensor_bytes(self, t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        """LRU eviction — caller must hold self._lock."""
        while self._used_bytes + incoming_bytes > self.capacity_bytes and self._order:
            oldest_key = self._order.pop(0)
            evicted = self._store.pop(oldest_key, None)
            if evicted is not None:
                self._used_bytes -= self._tensor_bytes(evicted)
                logger.debug("[DES-LOC][SLC] Evicted key=%s", oldest_key)

    # ------------------------------------------------------------------
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Store *tensor* in SLC (CPU DRAM, pin_memory for fast DMA).

        Args:
            key:    Unique identifier, e.g. "layer3/kv/step42".
            tensor: Source tensor (any device).
        """
        # Move to pinned CPU memory for PCIe DMA efficiency
        cpu_tensor = tensor.detach().to(self.device, non_blocking=True)
        try:
            cpu_tensor = cpu_tensor.pin_memory()
        except RuntimeError:
            pass   # pin_memory may fail on non-CUDA builds; degrade gracefully

        nb = self._tensor_bytes(cpu_tensor)
        with self._lock:
            # Remove old entry for this key if exists
            if key in self._store:
                self._used_bytes -= self._tensor_bytes(self._store[key])
                self._order.remove(key)
            self._evict_if_needed(nb)
            self._store[key] = cpu_tensor
            self._order.append(key)
            self._used_bytes += nb
        logger.debug("[DES-LOC][SLC] PUT key=%s  shape=%s  used=%.1f MB",
                     key, list(tensor.shape), self._used_bytes / 1024**2)

    def get(
        self,
        key: str,
        target_device: Optional[torch.device] = None,
        non_blocking: bool = True,
    ) -> Optional[torch.Tensor]:
        """Retrieve tensor from SLC, optionally moving it to *target_device*.

        Args:
            key:           Key used in put().
            target_device: If given, tensor is moved here (e.g. H100 for core_attn).
            non_blocking:  Use async DMA transfer (default True for PCIe overlap).

        Returns:
            Tensor on *target_device*, or None if key not found.
        """
        with self._lock:
            t = self._store.get(key, None)
        if t is None:
            logger.debug("[DES-LOC][SLC] MISS key=%s", key)
            return None
        if target_device is not None and t.device != target_device:
            t = t.to(target_device, non_blocking=non_blocking)
        logger.debug("[DES-LOC][SLC] GET key=%s  target=%s", key, target_device)
        return t

    def invalidate(self, key: str) -> None:
        """Remove a single entry from SLC (e.g. after backward_dw completes)."""
        with self._lock:
            if key in self._store:
                self._used_bytes -= self._tensor_bytes(self._store[key])
                self._order.remove(key)
                del self._store[key]
                logger.debug("[DES-LOC][SLC] INVALIDATED key=%s", key)

    def stats(self) -> dict:
        with self._lock:
            return {
                "used_mb":     self._used_bytes / 1024**2,
                "capacity_mb": self.capacity_bytes / 1024**2,
                "num_entries": len(self._store),
            }


# ---------------------------------------------------------------------------
# Submodule dataclasses (mirrors Megatron SelfAttentionSubmodules)
# ---------------------------------------------------------------------------

@dataclass
class HeteroSelfAttentionSubmodules:
    """Builder-level spec for DES-LOC heterogeneous self-attention.

    上游对应: megatron/core/transformer/attention.py :: SelfAttentionSubmodules
    变化: linear_qkv / core_attention 字段类型从 Union[ModuleSpec, type]
    升级为对应的 Builder Protocol，与 Megatron commit 90e685b 的意图一致。

    DES-LOC 扩展:
      linear_proj 也接受 Builder，以便 proj_device 和 attn_device 解耦。
    """
    linear_qkv:    LinearQkvBuilder
    core_attention: CoreAttentionBuilder
    linear_proj:   Optional[Callable] = None
    q_layernorm:   Optional[Callable] = None
    k_layernorm:   Optional[Callable] = None


@dataclass
class HeteroCrossAttentionSubmodules:
    """Builder-level spec for DES-LOC heterogeneous cross-attention.

    上游对应: megatron/core/transformer/attention.py :: CrossAttentionSubmodules
    """
    linear_q:      "LinearLayerBuilder"
    linear_kv:     "LinearLayerBuilder"
    core_attention: CoreAttentionBuilder
    linear_proj:   Optional[Callable] = None


@runtime_checkable
class LinearLayerBuilder(Protocol):
    """Factory for linear_q / linear_kv (CrossAttention).

    上游: megatron/core/transformer/attention.py :: LinearLayerBuilder
    """
    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: HeteroTransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        device: Optional[torch.device] = None,
    ) -> "LinearLayer":
        ...


@runtime_checkable
class LinearLayer(Protocol):
    """Protocol for linear_q / linear_kv instances."""
    def forward(self, input: torch.Tensor, /) -> Tuple[torch.Tensor, object]:
        ...


# ---------------------------------------------------------------------------
# Baseline (non-TE) implementations for smoke testing
# ---------------------------------------------------------------------------

class _BaselineLinearQkv(nn.Module):
    """Minimal ColumnParallelLinear-like wrapper for single-device testing.

    DES-LOC: In production, replace with a TE or Kitchen backend that
    supports delay_wgrad_compute and runs on proj_device.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: HeteroTransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool = False,
        bias: bool = False,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = "",
        tp_group=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.linear = nn.Linear(input_size, output_size, bias=bias, device=dev)
        if init_method is not None:
            with torch.no_grad():
                init_method(self.linear.weight)
        self._wgrad_pending = False
        logger.debug(
            "[DES-LOC][BaselineLinearQkv] in=%d out=%d bias=%s device=%s",
            input_size, output_size, bias, dev,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        out = self.linear(input)
        self._wgrad_pending = True
        return out, None

    def backward_dw(self) -> None:
        """No-op in baseline; hooks into delay_wgrad_compute pipeline in TE backend."""
        self._wgrad_pending = False
        logger.debug("[DES-LOC][BaselineLinearQkv] backward_dw() called (no-op in baseline)")


class _BaselineCoreAttention(nn.Module):
    """Scaled dot-product attention — runs on attn_device.

    DES-LOC: On H100 this should be replaced by FlashAttention-3 via TE.
    For SM86 A6000 fallback, FlashAttention-2 is used instead.
    """

    def __init__(
        self,
        *,
        config: HeteroTransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str],
        softmax_scale: Optional[float],
        pg_collection: Optional[object],
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.device = device or torch.device("cpu")
        scale = softmax_scale or (config.kv_channels ** -0.5)
        self.scale = scale
        logger.debug(
            "[DES-LOC][BaselineCoreAttn] layer=%d  mask=%s  device=%s  scale=%.4f",
            layer_number, attn_mask_type, self.device, scale,
        )

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        /,
        *,
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor],
        packed_seq_params: Optional[object],
    ) -> torch.Tensor:
        # Move Q/K/V to attn_device if necessary (PCIe transfer via SLC already done upstream)
        q = query.to(self.device, non_blocking=True)
        k = key.to(self.device, non_blocking=True)
        v = value.to(self.device, non_blocking=True)

        # [sq, b, np, hn] → [b, np, sq, hn]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask_type == AttnMaskType.causal:
            sq = scores.size(-2)
            sk = scores.size(-1)
            causal_mask = torch.ones(sq, sk, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            m = attention_mask.to(scores.device)
            scores = scores + m

        if attention_bias is not None:
            scores = scores + attention_bias.to(scores.device)

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)    # [b, np, sq, hn]
        # [b, np, sq, hn] → [sq, b, np*hn]
        b, np_, sq, hn = context.shape
        context = context.permute(2, 0, 1, 3).contiguous().view(sq, b, np_ * hn)
        return context


# ---------------------------------------------------------------------------
# HeteroAttentionProtocol  — main DES-LOC self-attention
# ---------------------------------------------------------------------------

class HeteroAttentionProtocol(nn.Module):
    """DES-LOC Heterogeneous Self-Attention.

    上游对应: megatron/core/transformer/attention.py :: SelfAttention
    关键适配:
      * linear_qkv 建在 proj_device (A6000)
      * core_attention 建在 attn_device (H100)
      * Q/K/V 经 SharedLocalityCache (SLC) 流动：
          proj_device → SLC(CPU DRAM) → attn_device
      * backward_dw() 遵循 delay_wgrad_compute 协议，
        在 H100 backward 完成后由调度器显式调用

    设备间数据流 (forward):
        A6000 ──linear_qkv──▶ Q,K,V (A6000)
                               │
                          SLC.put()  (pin_memory → CPU DRAM)
                               │
                          SLC.get(target=H100)  (async DMA)
                               │
        H100  ──core_attn──▶ context (H100)
                               │
                          .to(proj_device)  (结果搬回 A6000 做 linear_proj)
    """

    def __init__(
        self,
        config: HeteroTransformerConfig,
        submodules: HeteroSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
        policy: Optional[DevicePlacementPolicy] = None,
        slc: Optional[SharedLocalityCache] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.policy = policy or DevicePlacementPolicy.auto_detect()
        self.slc = slc or SharedLocalityCache(
            capacity_mb=config.slc_capacity_mb,
            device=self.policy.slc_device,
        )

        # Derived dimensions (mirrors Megatron Attention.__init__)
        assert config.kv_channels is not None
        assert config.num_query_groups is not None
        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size    = config.kv_channels * config.num_query_groups
        self.linear_qkv_out_dim    = self.query_projection_size + 2 * self.kv_projection_size
        if config.attention_output_gate:
            self.linear_qkv_out_dim += config.kv_channels * config.num_attention_heads

        init_fn = not_none(config.init_method) if config.init_method is not None else (
            lambda w: nn.init.xavier_uniform_(w)
        )

        # Build linear_qkv on proj_device (A6000)
        logger.info(
            "[DES-LOC][HeteroSelfAttn] layer=%d  Building linear_qkv on %s",
            layer_number, self.policy.proj_device,
        )
        self.linear_qkv: LinearQkv = submodules.linear_qkv(
            config.hidden_size,
            self.linear_qkv_out_dim,
            config=config,
            init_method=init_fn,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            device=self.policy.proj_device,
        )

        # Build core_attention on attn_device (H100)
        logger.info(
            "[DES-LOC][HeteroSelfAttn] layer=%d  Building core_attention on %s",
            layer_number, self.policy.attn_device,
        )
        self.core_attention: CoreAttention = submodules.core_attention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            softmax_scale=None,
            pg_collection=pg_collection,
            device=self.policy.attn_device,
        )

        # Optional projection layer (stays on proj_device)
        self.linear_proj: Optional[nn.Module] = None
        if submodules.linear_proj is not None:
            self.linear_proj = submodules.linear_proj(
                self.query_projection_size,
                config.hidden_size,
                config=config,
                init_method=init_fn,
                device=self.policy.proj_device,
            )

        # Q/K layer norms (optional, on proj_device)
        self.q_layernorm: Optional[nn.Module] = None
        self.k_layernorm: Optional[nn.Module] = None
        if submodules.q_layernorm is not None:
            self.q_layernorm = submodules.q_layernorm(config.kv_channels)
        if submodules.k_layernorm is not None:
            self.k_layernorm = submodules.k_layernorm(config.kv_channels)

        logger.info(
            "[DES-LOC][HeteroSelfAttn] layer=%d  Initialized  "
            "proj_device=%s  attn_device=%s  slc=%.0fMB",
            layer_number, self.policy.proj_device, self.policy.attn_device,
            config.slc_capacity_mb,
        )

    # ------------------------------------------------------------------
    def _slc_key(self, step: int, tensor_name: str) -> str:
        return f"layer{self.layer_number}_{tensor_name}_step{step}"

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Heterogeneous forward pass.

        Args:
            hidden_states:    [seq_len, batch, hidden] on proj_device.
            attention_mask:   Optional mask tensor.
            step:             Global training step (used as SLC key suffix).
            attention_bias:   Optional additive bias for attention scores.
            packed_seq_params: Variable-length sequence metadata.

        Returns:
            (output, None)  where output is [seq_len, batch, hidden] on proj_device.
        """
        # ── Stage 1: QKV projection on A6000 ──────────────────────────────────
        hs = hidden_states.to(self.policy.proj_device, non_blocking=True)
        mixed_qkv, _ = self.linear_qkv.forward(hs)   # [sq, b, qkv_dim]

        # Split Q / K / V
        q_size  = self.query_projection_size
        kv_size = self.kv_projection_size
        query  = mixed_qkv[..., :q_size]
        key    = mixed_qkv[..., q_size:q_size + kv_size]
        value  = mixed_qkv[..., q_size + kv_size:]

        # Reshape to multi-head: [sq, b, np, hn]
        sq, b = query.shape[:2]
        hn  = self.config.kv_channels
        np_ = self.config.num_attention_heads
        ng  = self.config.num_query_groups
        query  = query.view(sq, b, np_,  hn)
        key    = key.view(sq,   b, ng,   hn)
        value  = value.view(sq, b, ng,   hn)

        # Optional Q/K layer norms
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        # ── Stage 2: SLC staging — A6000 → CPU DRAM ───────────────────────────
        # Stash Q/K/V in SLC so that:
        #  a) H100 can DMA-prefetch while A6000 still holds the tensor alive
        #  b) backward pass can retrieve KV without re-computing
        self.slc.put(self._slc_key(step, "query"), query)
        self.slc.put(self._slc_key(step, "key"),   key)
        self.slc.put(self._slc_key(step, "value"), value)

        # ── Stage 3: Retrieve on H100 (async DMA, PCIe) ───────────────────────
        q_h100 = self.slc.get(self._slc_key(step, "query"), target_device=self.policy.attn_device)
        k_h100 = self.slc.get(self._slc_key(step, "key"),   target_device=self.policy.attn_device)
        v_h100 = self.slc.get(self._slc_key(step, "value"), target_device=self.policy.attn_device)

        if q_h100 is None or k_h100 is None or v_h100 is None:
            # SLC evicted the tensor before we could retrieve it (capacity too small)
            # Fall back to direct device-to-device transfer
            logger.warning(
                "[DES-LOC][HeteroSelfAttn] SLC miss on layer=%d step=%d; "
                "falling back to direct P2P transfer.",
                self.layer_number, step,
            )
            q_h100 = query.to(self.policy.attn_device, non_blocking=False)
            k_h100 = key.to(self.policy.attn_device,   non_blocking=False)
            v_h100 = value.to(self.policy.attn_device, non_blocking=False)

        mask_h100 = (
            attention_mask.to(self.policy.attn_device, non_blocking=True)
            if attention_mask is not None else None
        )

        # ── Stage 4: Core attention on H100 ───────────────────────────────────
        context = self.core_attention.forward(
            q_h100, k_h100, v_h100, mask_h100,
            attn_mask_type=self.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )   # [sq, b, hidden] on H100

        # ── Stage 5: Move context back to proj_device for linear_proj ─────────
        context_proj = context.to(self.policy.proj_device, non_blocking=True)

        if self.linear_proj is not None:
            output, _ = self.linear_proj(context_proj)
        else:
            output = context_proj

        return output, None

    # ------------------------------------------------------------------
    def backward_dw(self) -> None:
        """Trigger delayed weight gradient for linear_qkv.

        DES-LOC schedule:
          1. Engine calls backward() — activation grads flow through core_attn (H100)
             and then through linear_qkv (A6000).
          2. Engine calls backward_dw() — weight grad all-reduce is deferred here,
             overlapping with H100's next forward step.
          3. SLC entries for this layer/step are invalidated.

        对应 Megatron SelfAttention.backward_dw() 在 delay_wgrad_compute=True 路径。
        """
        logger.debug("[DES-LOC][HeteroSelfAttn] layer=%d backward_dw()", self.layer_number)
        if hasattr(self.linear_qkv, "backward_dw"):
            self.linear_qkv.backward_dw()


# ---------------------------------------------------------------------------
# HeteroCrossAttentionProtocol
# ---------------------------------------------------------------------------

class HeteroCrossAttentionProtocol(nn.Module):
    """DES-LOC Heterogeneous Cross-Attention.

    上游对应: megatron/core/transformer/attention.py :: CrossAttention
    DES-LOC 说明:
      linear_q (query projection from decoder hidden states) → proj_device
      linear_kv (key/value projection from encoder output)  → proj_device
      core_attention                                         → attn_device
      KV cache 通过 SLC 跨 step 复用，避免 encoder KV 在每个 decoder step 重算。
    """

    def __init__(
        self,
        config: HeteroTransformerConfig,
        submodules: HeteroCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
        policy: Optional[DevicePlacementPolicy] = None,
        slc: Optional[SharedLocalityCache] = None,
    ):
        super().__init__()
        self.config        = config
        self.layer_number  = layer_number
        self.attn_mask_type = attn_mask_type
        self.policy = policy or DevicePlacementPolicy.auto_detect()
        self.slc = slc or SharedLocalityCache(
            capacity_mb=config.slc_capacity_mb,
            device=self.policy.slc_device,
        )

        assert config.kv_channels is not None
        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size    = config.kv_channels * config.num_attention_heads

        init_fn = config.init_method or (lambda w: nn.init.xavier_uniform_(w))

        self.linear_q: LinearLayer = submodules.linear_q(
            config.hidden_size,
            self.query_projection_size,
            config=config,
            init_method=not_none(init_fn),
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            device=self.policy.proj_device,
        )

        self.linear_kv: LinearLayer = submodules.linear_kv(
            config.hidden_size,
            2 * self.kv_projection_size,
            config=config,
            init_method=not_none(init_fn),
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            device=self.policy.proj_device,
        )

        self.core_attention: CoreAttention = submodules.core_attention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
            softmax_scale=None,
            pg_collection=pg_collection,
            device=self.policy.attn_device,
        )

        logger.info(
            "[DES-LOC][HeteroCrossAttn] layer=%d  proj_device=%s  attn_device=%s",
            layer_number, self.policy.proj_device, self.policy.attn_device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, None]:
        """Heterogeneous cross-attention forward.

        Args:
            hidden_states:    Decoder hidden states [sq, b, hidden].
            key_value_states: Encoder output [sk, b, hidden].
            attention_mask:   Optional mask.
            step:             Training step for SLC key.

        Returns:
            (context, None)
        """
        assert key_value_states is not None, (
            "[DES-LOC] key_value_states cannot be None for CrossAttention"
        )

        # KV projection — try SLC first (encoder KV may be cached from prev step)
        kv_cache_key = f"layer{self.layer_number}_kv_enc_step{step}"
        cached_kv = self.slc.get(kv_cache_key, target_device=self.policy.proj_device)

        if cached_kv is None:
            enc = key_value_states.to(self.policy.proj_device, non_blocking=True)
            mixed_kv, _ = self.linear_kv.forward(enc)
            self.slc.put(kv_cache_key, mixed_kv)
        else:
            mixed_kv = cached_kv
            logger.debug(
                "[DES-LOC][HeteroCrossAttn] KV cache HIT layer=%d step=%d",
                self.layer_number, step,
            )

        dec = hidden_states.to(self.policy.proj_device, non_blocking=True)
        query, _ = self.linear_q.forward(dec)

        # Split KV: [sk, b, 2*hn] → key [sk,b,hn], value [sk,b,hn]
        sk, b = mixed_kv.shape[:2]
        hn  = self.config.kv_channels
        np_ = self.config.num_attention_heads
        mixed_kv_r = mixed_kv.view(sk, b, np_, 2 * hn)
        key   = mixed_kv_r[..., :hn].contiguous()
        value = mixed_kv_r[..., hn:].contiguous()

        sq = query.shape[0]
        query = query.view(sq, b, np_, hn)

        # SLC transfer to H100
        self.slc.put(f"layer{self.layer_number}_q_step{step}",   query)
        self.slc.put(f"layer{self.layer_number}_key_step{step}",  key)
        self.slc.put(f"layer{self.layer_number}_val_step{step}",  value)

        q_h = self.slc.get(f"layer{self.layer_number}_q_step{step}",   target_device=self.policy.attn_device) or query.to(self.policy.attn_device)
        k_h = self.slc.get(f"layer{self.layer_number}_key_step{step}", target_device=self.policy.attn_device) or key.to(self.policy.attn_device)
        v_h = self.slc.get(f"layer{self.layer_number}_val_step{step}", target_device=self.policy.attn_device) or value.to(self.policy.attn_device)

        mask_h = attention_mask.to(self.policy.attn_device) if attention_mask is not None else None

        context = self.core_attention.forward(
            q_h, k_h, v_h, mask_h,
            attn_mask_type=self.attn_mask_type,
            attention_bias=None,
            packed_seq_params=None,
        )

        return context.to(self.policy.proj_device, non_blocking=True), None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_hetero_self_attention_spec(
    config: HeteroTransformerConfig,
    use_causal_mask: bool = True,
) -> Tuple[HeteroSelfAttentionSubmodules, DevicePlacementPolicy]:
    """Construct a spec + policy pair for DES-LOC self-attention.

    Returns the submodule spec (with baseline builders) and an auto-detected
    DevicePlacementPolicy.  In production, swap the baseline builders for TE
    or Kitchen backends targeting SM90 / SM86 respectively.

    Args:
        config:          HeteroTransformerConfig instance.
        use_causal_mask: If True, use causal masking (GPT-style); else padding.

    Returns:
        (submodules, policy)
    """
    policy = DevicePlacementPolicy.auto_detect()
    mask   = AttnMaskType.causal if use_causal_mask else AttnMaskType.padding

    def _linear_qkv_builder(
        input_size: int, output_size: int, /, *,
        config: HeteroTransformerConfig,
        init_method,
        gather_output, bias, skip_bias_add,
        is_expert, tp_comm_buffer_name,
        tp_group=None,
        device=None,
    ) -> _BaselineLinearQkv:
        return _BaselineLinearQkv(
            input_size, output_size,
            config=config, init_method=init_method,
            gather_output=gather_output, bias=bias,
            skip_bias_add=skip_bias_add, is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group, device=device,
        )

    def _core_attn_builder(
        *, config, layer_number, attn_mask_type, attention_type,
        cp_comm_type, softmax_scale, pg_collection, device=None,
    ) -> _BaselineCoreAttention:
        return _BaselineCoreAttention(
            config=config, layer_number=layer_number,
            attn_mask_type=attn_mask_type, attention_type=attention_type,
            cp_comm_type=cp_comm_type, softmax_scale=softmax_scale,
            pg_collection=pg_collection, device=device,
        )

    spec = HeteroSelfAttentionSubmodules(
        linear_qkv=_linear_qkv_builder,
        core_attention=_core_attn_builder,
    )
    return spec, policy


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ── Test 1: not_none raises on None ──────────────────────────────────────
    try:
        not_none(None)
        assert False, "Should have raised"
    except ValueError:
        pass
    assert not_none(42) == 42

    # ── Test 2: SLC put/get/invalidate ───────────────────────────────────────
    slc = SharedLocalityCache(capacity_mb=1.0, device=torch.device("cpu"))
    t = torch.randn(4, 4)
    slc.put("test_key", t)
    retrieved = slc.get("test_key")
    assert retrieved is not None and retrieved.shape == t.shape
    slc.invalidate("test_key")
    assert slc.get("test_key") is None

    # ── Test 3: DevicePlacementPolicy CPU fallback ───────────────────────────
    if not torch.cuda.is_available():
        policy = DevicePlacementPolicy.auto_detect()
        assert policy.proj_device.type == "cpu"
        assert policy.slc_device.type  == "cpu"

    # ── Test 4: HeteroAttentionProtocol forward (CPU, baseline builders) ─────
    cfg = HeteroTransformerConfig(
        hidden_size=64, num_attention_heads=4, num_query_groups=4,
        kv_channels=16, slc_capacity_mb=16.0,
    )
    spec, policy = build_hetero_self_attention_spec(cfg, use_causal_mask=True)
    # Override policy to CPU for portability
    policy = DevicePlacementPolicy(
        proj_device=torch.device("cpu"),
        attn_device=torch.device("cpu"),
        slc_device=torch.device("cpu"),
    )
    attn = HeteroAttentionProtocol(cfg, spec, layer_number=1, policy=policy)
    dummy = torch.randn(8, 2, 64)   # [seq_len=8, batch=2, hidden=64]
    out, bias = attn.forward(dummy, step=0)
    assert out.shape == dummy.shape, f"Expected {dummy.shape}, got {out.shape}"

    # ── Test 5: backward_dw does not raise ───────────────────────────────────
    attn.backward_dw()

    print("All smoke tests passed.")
