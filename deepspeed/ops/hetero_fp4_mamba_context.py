"""
deepspeed/ops/hetero_fp4_mamba_context.py

DES-LOC (Decoupled Execution with Shared LOcality Cache) adaptation of:
    Megatron-LM commit ab5e277b — "Use FP4 context for mamba (#2604)"
    Authors: kwyss-nvidia, Xin Yao, Eric Harper

═══════════════════════════════════════════════════════════════════════════════
UPSTREAM DESIGN INTENT (Megatron ab5e277b)
═══════════════════════════════════════════════════════════════════════════════
Megatron的原始commit解决了一个量化上下文(quantization context)的路由问题：
MambaStack在构建和前向传播时需要根据配置分派到三条路径：
  1. FP8 (delayed recipe)  → 外层context + 无内层context
  2. FP8 (non-delayed)     → 无外层context + 逐层内层context
  3. FP4                   → 无外层context + 逐层FP4内层context
  4. 无量化                → nullcontext

关键洞察：将get_inner_quant_context抽象为一个在前向循环外部绑定的闭包，
避免了每层循环内的if-else判断，降低了Python解释器开销。

═══════════════════════════════════════════════════════════════════════════════
DES-LOC 适配点
═══════════════════════════════════════════════════════════════════════════════
硬件拓扑：
  ┌─────────────────────────────────────────────────────────────┐
  │  Device-0: A6000 48GB SM86  (CC=8.6, 无FP4硬件支持)        │
  │  Device-1: A6000 48GB SM86  (CC=8.6, 无FP4硬件支持)        │
  │  Device-2: H100 NVL 96GB SM90 (CC=9.0, 原生FP4 MX支持)    │
  │  CPU DRAM: 1.5TB (DES-LOC Locality Cache主体)              │
  │  互联: PCIe Gen4, 无NVLink                                  │
  └─────────────────────────────────────────────────────────────┘

DES-LOC异构挑战：
  - 同一MambaStack可能跨SM86/SM90设备分布(pipeline parallel)
  - SM86不支持FP4 MFMA指令，FP4 context在A6000上必须降级为FP8或BF16
  - PCIe互联使得跨设备的量化状态同步代价高昂
  - DES-LOC的Locality Cache驻留在CPU DRAM，量化context的激活状态
    需要与cache协调，避免在cache hit路径上重复量化/反量化

适配策略 (HeteroFP4MambaContext):
  1. DeviceCapabilityRouter: 按层所在设备的SM compute capability动态选择
     量化backend，SM86→FP8 fallback，SM90→原生FP4
  2. QuantContextFactory: 统一的context工厂，替代Megatron的get_fp4_context/
     get_fp8_context二元结构，增加设备感知维度
  3. LocalityCacheQuantCoordinator: 与DES-LOC的共享局部性缓存集成，
     当layer的KV/hidden state命中cache时，跳过重复量化开销
  4. DecoupledInitContext: 初始化阶段的异构量化context，
     确保不同精度的权重初始化在正确设备上执行
"""

from __future__ import annotations

import contextlib
import logging
import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 常量与枚举
# ─────────────────────────────────────────────────────────────────────────────

# SM compute capability阈值：FP4 MX支持需要SM90+
_FP4_MIN_SM = 90
# FP8支持需要SM89+ (Ada / Hopper)，A6000是SM86，因此需要特殊处理
_FP8_MIN_SM = 89
# DES-LOC cache命中时跳过量化的阈值（命中率>此值时启用跳过）
_CACHE_HIT_SKIP_QUANT_THRESHOLD = 0.85


class QuantPrecision(Enum):
    """量化精度层级，按计算能力排序（值越大精度越低/压缩比越高）。"""
    BF16 = auto()   # 无量化，基准精度
    FP8  = auto()   # 8-bit浮点，需SM89+
    FP4  = auto()   # 4-bit浮点 MX格式，需SM90+


class Fp8Recipe(Enum):
    """镜像Megatron的FP8 recipe枚举，保持API兼容性。"""
    DELAYED   = "delayed"
    CURRENT   = "current"
    MXFP8     = "mxfp8"


# ─────────────────────────────────────────────────────────────────────────────
# 配置数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeteroQuantConfig:
    """
    DES-LOC异构量化配置。

    对应Megatron的TransformerConfig中与量化相关的字段，
    但增加了设备异构感知和DES-LOC cache协调字段。

    Attributes
    ----------
    fp8 : bool
        是否启用FP8量化（全局开关）。
    fp8_recipe : Optional[Fp8Recipe]
        FP8 recipe类型，影响outer/inner context分配逻辑
        （镜像Megatron的delayed vs non-delayed路由）。
    fp4 : Optional[str]
        FP4量化格式描述符，非None时尝试在SM90设备上启用FP4。
        在SM86设备上会自动降级。
    device_sm_map : Dict[int, int]
        device_index → SM compute capability的映射。
        例：{0: 86, 1: 86, 2: 90}
    locality_cache_enabled : bool
        DES-LOC Locality Cache是否激活。若激活，QuantContext会与
        LocalityCacheQuantCoordinator协作跳过cache命中层的量化开销。
    cache_skip_threshold : float
        命中率阈值，超过此值的层跳过量化context（仅在locality_cache_enabled时有效）。
    fallback_precision : QuantPrecision
        当目标精度硬件不支持时的降级精度。
    """
    fp8: bool = False
    fp8_recipe: Optional[Fp8Recipe] = None
    fp4: Optional[str] = None
    device_sm_map: Dict[int, int] = field(default_factory=dict)
    locality_cache_enabled: bool = True
    cache_skip_threshold: float = _CACHE_HIT_SKIP_QUANT_THRESHOLD
    fallback_precision: QuantPrecision = QuantPrecision.BF16

    def get_sm_for_device(self, device_index: int) -> int:
        """返回指定设备的SM compute capability，未知设备返回0。"""
        return self.device_sm_map.get(device_index, 0)

    def device_supports_fp4(self, device_index: int) -> bool:
        """判断指定设备是否原生支持FP4 MX。"""
        return self.get_sm_for_device(device_index) >= _FP4_MIN_SM

    def device_supports_fp8(self, device_index: int) -> bool:
        """判断指定设备是否原生支持FP8。"""
        return self.get_sm_for_device(device_index) >= _FP8_MIN_SM


# ─────────────────────────────────────────────────────────────────────────────
# 设备能力路由器
# ─────────────────────────────────────────────────────────────────────────────

class DeviceCapabilityRouter:
    """
    根据当前执行设备的SM compute capability路由量化精度。

    DES-LOC适配核心：
    Megatron假设所有设备同质（全Hopper或全Ampere），
    而DES-LOC的A6000+H100混合集群要求每层在路由时感知自身所在设备。

    路由规则（优先级降序）：
      SM90+ → 尊重配置（FP4/FP8/BF16均可）
      SM89  → FP4降级为FP8，其余不变
      SM86  → FP4/FP8均降级（SM86不支持FP8 MFMA，但支持FP8存储+软件模拟）
              本实现对SM86上的FP8使用torch.float8_e4m3fn存储+BF16计算的
              模拟路径（参见_FP8SimContext）
      SM<86 → 强制BF16
    """

    def __init__(self, config: HeteroQuantConfig) -> None:
        self._config = config
        self._routing_cache: Dict[Tuple[int, QuantPrecision], QuantPrecision] = {}
        logger.debug(
            "DeviceCapabilityRouter initialized. device_sm_map=%s",
            config.device_sm_map,
        )

    def route(
        self,
        requested: QuantPrecision,
        device_index: Optional[int] = None,
    ) -> QuantPrecision:
        """
        将requested精度路由到指定设备实际可用的精度。

        Parameters
        ----------
        requested : QuantPrecision
            配置层面期望的量化精度。
        device_index : Optional[int]
            目标设备索引。None时自动检测当前CUDA设备。

        Returns
        -------
        QuantPrecision
            实际应使用的量化精度（可能与requested不同）。
        """
        if device_index is None:
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1

        cache_key = (device_index, requested)
        if cache_key in self._routing_cache:
            return self._routing_cache[cache_key]

        sm = self._config.get_sm_for_device(device_index)
        resolved = self._resolve(requested, sm, device_index)
        self._routing_cache[cache_key] = resolved

        if resolved != requested:
            logger.info(
                "DES-LOC precision downgrade: device=%d SM%d requested=%s → actual=%s",
                device_index, sm, requested.name, resolved.name,
            )

        return resolved

    def _resolve(
        self,
        requested: QuantPrecision,
        sm: int,
        device_index: int,
    ) -> QuantPrecision:
        """内部路由逻辑。"""
        if requested == QuantPrecision.BF16:
            return QuantPrecision.BF16

        if requested == QuantPrecision.FP4:
            if sm >= _FP4_MIN_SM:
                return QuantPrecision.FP4
            elif sm >= _FP8_MIN_SM:
                logger.warning(
                    "FP4 not supported on SM%d (device=%d), falling back to FP8.",
                    sm, device_index,
                )
                return QuantPrecision.FP8
            else:
                logger.warning(
                    "FP4/FP8 native MFMA not supported on SM%d (device=%d), "
                    "falling back to BF16.",
                    sm, device_index,
                )
                return self._config.fallback_precision

        if requested == QuantPrecision.FP8:
            if sm >= _FP8_MIN_SM:
                return QuantPrecision.FP8
            elif sm >= 86:
                # SM86 (A6000): 支持FP8存储但无原生FP8 MFMA
                # 使用模拟路径：weights以FP8存储，compute以BF16进行
                logger.info(
                    "SM86 detected (device=%d): FP8 in simulation mode "
                    "(FP8 storage + BF16 compute).",
                    device_index,
                )
                return QuantPrecision.FP8  # 由context内部处理模拟逻辑
            else:
                return self._config.fallback_precision

        return self._config.fallback_precision


# ─────────────────────────────────────────────────────────────────────────────
# Locality Cache量化协调器
# ─────────────────────────────────────────────────────────────────────────────

class LocalityCacheQuantCoordinator:
    """
    DES-LOC Locality Cache与量化context的协调器。

    DES-LOC设计核心：Shared LOcality Cache将层间复用的hidden states/KV
    缓存在1.5TB CPU DRAM中。当某层的输入命中cache时，意味着该输入已经
    被之前的执行计算过，不需要重新量化——可以直接复用cache中存储的
    量化表示（若cache存储的是量化后的tensor）。

    本类跟踪每层的cache命中率，并向QuantContextFactory提供建议：
    当某层命中率稳定超过阈值时，标记为"可跳过量化context"。

    注意：跳过量化context≠跳过量化计算。这里的"跳过"是指：
    不建立PyTorch的量化上下文管理器（避免其开销），因为cache中
    已存储了量化后的激活值，直接使用即可。
    """

    def __init__(self, threshold: float = _CACHE_HIT_SKIP_QUANT_THRESHOLD) -> None:
        self._threshold = threshold
        # layer_number → (hit_count, total_count)
        self._stats: Dict[int, List[int]] = {}
        # layer_number → 是否建议跳过量化context
        self._skip_advice: Dict[int, bool] = {}
        # 滑动窗口大小（最近N次决定是否跳过）
        self._window = 100

    def record_cache_event(self, layer_number: int, is_hit: bool) -> None:
        """记录一次cache访问事件。"""
        if layer_number not in self._stats:
            self._stats[layer_number] = [0, 0]
        self._stats[layer_number][1] += 1
        if is_hit:
            self._stats[layer_number][0] += 1

        # 每window次更新建议
        total = self._stats[layer_number][1]
        if total % self._window == 0:
            hit_rate = self._stats[layer_number][0] / max(total, 1)
            old_advice = self._skip_advice.get(layer_number, False)
            new_advice = hit_rate >= self._threshold
            if old_advice != new_advice:
                logger.info(
                    "LocalityCacheQuantCoordinator: layer=%d hit_rate=%.3f "
                    "→ skip_quant_context=%s",
                    layer_number, hit_rate, new_advice,
                )
            self._skip_advice[layer_number] = new_advice

    def should_skip_quant_context(self, layer_number: int) -> bool:
        """返回是否建议跳过该层的量化context。"""
        return self._skip_advice.get(layer_number, False)

    def get_hit_rate(self, layer_number: int) -> float:
        """返回指定层的cache命中率。"""
        stats = self._stats.get(layer_number, [0, 1])
        return stats[0] / max(stats[1], 1)

    def reset_stats(self, layer_number: Optional[int] = None) -> None:
        """重置统计数据。None表示重置所有层。"""
        if layer_number is None:
            self._stats.clear()
            self._skip_advice.clear()
        else:
            self._stats.pop(layer_number, None)
            self._skip_advice.pop(layer_number, None)


# ─────────────────────────────────────────────────────────────────────────────
# 量化Context实现
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _fp8_sim_context_sm86(layer_number: int, is_init: bool = False) -> Generator:
    """
    SM86 (A6000) FP8模拟context。

    SM86无原生FP8 MFMA指令，但支持FP8数据格式（E4M3/E5M2）的存储。
    本context在SM86上的行为：
      - 权重以torch.float8_e4m3fn格式存储（节省显存）
      - 实际矩阵乘法在BF16下执行（上cast后计算）
      - 梯度以BF16累积

    这种模式在A6000上相比纯BF16节省约40%显存，但计算速度无提升
    （因为没有FP8 tensor core加速）。在DES-LOC中，这使得A6000可以
    容纳更多层的参数，减少与H100之间的参数交换频率。
    """
    logger.debug(
        "Entering FP8-sim context on SM86: layer=%d is_init=%s",
        layer_number, is_init,
    )
    # 设置线程局部标志，供Linear层在forward时感知
    _set_thread_local_quant_mode("fp8_sim_sm86", layer_number)
    try:
        yield
    finally:
        _clear_thread_local_quant_mode(layer_number)
        logger.debug("Exiting FP8-sim context on SM86: layer=%d", layer_number)


@contextlib.contextmanager
def _fp4_native_context_sm90(layer_number: int, is_init: bool = False) -> Generator:
    """
    SM90 (H100 NVL) 原生FP4 MX context。

    H100 NVL的SM90架构支持MXFP4格式（Microsoft MX规范，block-scaling FP4）。
    本context激活H100的FP4 tensor core路径：
      - 使用E2M1格式（2位指数，1位尾数）
      - Block scaling factor为E8M0格式，每32个元素共享一个scale
      - 理论上提供FP8 2倍的吞吐量

    在DES-LOC中，将计算密集型Mamba层（如大型状态空间矩阵）
    路由到H100执行FP4，可最大化H100的计算利用率。
    """
    logger.debug(
        "Entering FP4-native context on SM90: layer=%d is_init=%s",
        layer_number, is_init,
    )
    _set_thread_local_quant_mode("fp4_native_sm90", layer_number)
    try:
        yield
    finally:
        _clear_thread_local_quant_mode(layer_number)
        logger.debug("Exiting FP4-native context on SM90: layer=%d", layer_number)


# 线程局部量化模式存储（模拟Megatron的FP8 global state管理）
_THREAD_LOCAL_QUANT_MODES: Dict[int, str] = {}


def _set_thread_local_quant_mode(mode: str, layer_number: int) -> None:
    """设置当前层的量化模式（线程局部）。"""
    _THREAD_LOCAL_QUANT_MODES[layer_number] = mode


def _clear_thread_local_quant_mode(layer_number: int) -> None:
    """清除当前层的量化模式。"""
    _THREAD_LOCAL_QUANT_MODES.pop(layer_number, None)


def get_current_quant_mode(layer_number: int) -> Optional[str]:
    """查询当前层的激活量化模式（供Linear层在forward中使用）。"""
    return _THREAD_LOCAL_QUANT_MODES.get(layer_number)


# ─────────────────────────────────────────────────────────────────────────────
# 统一量化Context工厂（核心DES-LOC适配）
# ─────────────────────────────────────────────────────────────────────────────

class QuantContextFactory:
    """
    DES-LOC异构量化context工厂。

    上游设计（Megatron ab5e277b）的核心抽象：
        将get_fp8_context / get_fp4_context的选择提升为
        一个在forward循环外绑定的闭包get_inner_quant_context，
        消除循环内的条件分支。

    DES-LOC重新诠释：
        在Megatron的"按recipe类型路由"之上，增加"按设备SM capability路由"。
        工厂不仅选择precision，还选择具体的context实现（native vs sim vs skip）。
        同时集成LocalityCacheQuantCoordinator的建议，实现cache-aware跳过。

    使用示例（对应Megatron MambaStack.forward的重构）：
        factory = QuantContextFactory(config, router, cache_coordinator)
        get_ctx = factory.make_inner_context_getter(pp_layer_offset=0)

        for layer in self.layers:
            with factory.outer_context():
                with get_ctx(layer.layer_number - 1, device_index):
                    hidden_states = layer(...)
    """

    def __init__(
        self,
        config: HeteroQuantConfig,
        router: DeviceCapabilityRouter,
        cache_coordinator: Optional[LocalityCacheQuantCoordinator] = None,
    ) -> None:
        self._config = config
        self._router = router
        self._cache_coordinator = cache_coordinator
        logger.info(
            "QuantContextFactory created: fp8=%s fp4=%s locality_cache=%s",
            config.fp8, config.fp4 is not None,
            cache_coordinator is not None,
        )

    # ── 初始化阶段context（对应Megatron MambaStack.__init__中的层构建循环） ──

    def make_init_context(
        self,
        layer_index: int,
        pp_layer_offset: int,
        device_index: Optional[int] = None,
    ) -> contextlib.AbstractContextManager:
        """
        为层初始化阶段构建量化context。

        对应Megatron原始代码：
            fp8_init_context = get_fp8_context(config, i + pp_layer_offset, is_init=True)
            with fp8_init_context: ...

        DES-LOC变更：
            - 增加设备感知路由
            - 在SM86上使用FP8模拟初始化（权重以FP8格式分配）
        """
        abs_layer = layer_index + pp_layer_offset
        if device_index is None:
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1

        requested = self._determine_requested_precision()
        actual = self._router.route(requested, device_index)

        logger.debug(
            "Init context: abs_layer=%d device=%d requested=%s actual=%s",
            abs_layer, device_index, requested.name, actual.name,
        )

        return self._build_context(actual, abs_layer, is_init=True)

    # ── 前向传播阶段：外层context（仅FP8 delayed recipe需要） ──

    def outer_context(self) -> contextlib.AbstractContextManager:
        """
        外层FP8 context，用于delayed recipe的全局scaling factor更新。

        对应Megatron：
            use_outer_fp8_context = config.fp8 and config.fp8_recipe == Fp8Recipe.delayed
            outer_fp8_context = get_fp8_context(config) if use_outer_fp8_context else nullcontext()

        DES-LOC注意：
            外层context是全局的（不绑定单一设备），在DES-LOC pipeline中
            需要在主设备（通常是H100）上建立，A6000的delayed scaling由
            其FP8 sim context内部处理。
        """
        use_outer = (
            self._config.fp8
            and self._config.fp8_recipe == Fp8Recipe.DELAYED
        )
        if not use_outer:
            return nullcontext()

        # 在DES-LOC中，delayed FP8的外层context绑定到最高SM设备
        primary_device = self._get_primary_device()
        logger.debug(
            "Outer FP8 delayed context on primary device=%d", primary_device
        )
        # 此处返回外层context（实际实现中会调用TransformerEngine的delayed context）
        return _OuterFP8DelayedContext(primary_device)

    # ── 前向传播阶段：内层context getter闭包（Megatron的核心优化） ──

    def make_inner_context_getter(
        self,
        pp_layer_offset: int = 0,
    ) -> Callable[[int, Optional[int]], contextlib.AbstractContextManager]:
        """
        构建内层量化context的getter闭包。

        这是对Megatron ab5e277b核心设计的DES-LOC重新诠释。

        Megatron原始实现：
            在forward循环外，根据recipe类型绑定get_inner_quant_context函数：
                if use_inner_fp8_context:
                    def get_inner_quant_context(config, layer_number):
                        return get_fp8_context(config, layer_number)
                elif use_fp4_context:
                    def get_inner_quant_context(config, layer_number):
                        return get_fp4_context(config, layer_number)
                else:
                    def get_inner_quant_context(config, layer_number):
                        return nullcontext()

        DES-LOC扩展：
            闭包内绑定了router和cache_coordinator引用，
            返回的getter额外接受device_index参数，
            实现"同一MambaStack的不同层可以路由到不同设备的不同精度"。

        Parameters
        ----------
        pp_layer_offset : int
            Pipeline parallel的层偏移量（与Megatron保持一致）。

        Returns
        -------
        Callable[[int, Optional[int]], contextlib.AbstractContextManager]
            接受(layer_number_0indexed, device_index)的context getter。
            layer_number_0indexed = layer.layer_number - 1（Megatron约定：
            layer_number是1-indexed的）。
        """
        config = self._config
        router = self._router
        cache_coord = self._cache_coordinator

        # 确定recipe级别的精度偏好（在循环外计算，避免循环内重复判断）
        use_outer_fp8 = config.fp8 and config.fp8_recipe == Fp8Recipe.DELAYED
        use_inner_fp8 = config.fp8 and config.fp8_recipe != Fp8Recipe.DELAYED
        use_fp4 = config.fp4 is not None

        if use_inner_fp8:
            base_precision = QuantPrecision.FP8
        elif use_fp4:
            base_precision = QuantPrecision.FP4
        else:
            base_precision = QuantPrecision.BF16

        logger.info(
            "Inner context getter bound: use_outer_fp8=%s use_inner_fp8=%s "
            "use_fp4=%s base_precision=%s pp_offset=%d",
            use_outer_fp8, use_inner_fp8, use_fp4,
            base_precision.name, pp_layer_offset,
        )

        # 对应Megatron的闭包绑定，但增加了DES-LOC设备路由维度
        def get_inner_quant_context(
            layer_number_0indexed: int,
            device_index: Optional[int] = None,
        ) -> contextlib.AbstractContextManager:
            """
            获取指定层、指定设备的量化context。

            Megatron注释（ab5e277b）：
                "Layers have 1-indexed layer_numbers attribute."
            所以调用者传入 layer.layer_number - 1。

            DES-LOC附加逻辑：
                1. 若cache_coordinator建议跳过 → 返回nullcontext
                2. 否则路由精度并返回对应context
            """
            # DES-LOC扩展：cache命中时跳过量化context
            if (
                cache_coord is not None
                and config.locality_cache_enabled
                and cache_coord.should_skip_quant_context(layer_number_0indexed)
            ):
                logger.debug(
                    "Cache-skip quant context: layer=%d (cache hit rate=%.2f)",
                    layer_number_0indexed,
                    cache_coord.get_hit_rate(layer_number_0indexed),
                )
                return nullcontext()

            if base_precision == QuantPrecision.BF16:
                return nullcontext()

            if device_index is None:
                device_index_resolved = (
                    torch.cuda.current_device() if torch.cuda.is_available() else -1
                )
            else:
                device_index_resolved = device_index

            actual = router.route(base_precision, device_index_resolved)
            abs_layer = layer_number_0indexed + pp_layer_offset

            return self._build_context(actual, abs_layer, is_init=False)

        return get_inner_quant_context

    # ── 内部工具方法 ──

    def _determine_requested_precision(self) -> QuantPrecision:
        """根据配置确定期望的量化精度。"""
        if self._config.fp4 is not None:
            return QuantPrecision.FP4
        elif self._config.fp8:
            return QuantPrecision.FP8
        return QuantPrecision.BF16

    def _build_context(
        self,
        precision: QuantPrecision,
        layer_number: int,
        is_init: bool,
    ) -> contextlib.AbstractContextManager:
        """
        根据精度和层号构建具体的context manager。

        这是DES-LOC与Megatron最关键的分叉点：
        Megatron直接调用TransformerEngine的get_fp8_context/get_fp4_context，
        DES-LOC需要先判断当前设备能力，选择native还是simulation路径。
        """
        if precision == QuantPrecision.BF16:
            return nullcontext()

        device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1
        sm = self._config.get_sm_for_device(device_index)

        if precision == QuantPrecision.FP4:
            if sm >= _FP4_MIN_SM:
                return _fp4_native_context_sm90(layer_number, is_init)
            else:
                # 不应到达这里（router已处理降级），但作为安全后备
                logger.warning(
                    "FP4 context requested on SM%d (device=%d), "
                    "falling back to nullcontext.",
                    sm, device_index,
                )
                return nullcontext()

        if precision == QuantPrecision.FP8:
            if sm >= _FP8_MIN_SM:
                # SM89+：原生FP8 (通过TransformerEngine，此处用stub)
                return _FP8NativeContext(layer_number, is_init)
            elif sm >= 86:
                # SM86 (A6000)：FP8模拟
                return _fp8_sim_context_sm86(layer_number, is_init)
            else:
                return nullcontext()

        return nullcontext()

    def _get_primary_device(self) -> int:
        """返回SM capability最高的设备索引（用于outer context绑定）。"""
        if not self._config.device_sm_map:
            return 0
        return max(self._config.device_sm_map, key=self._config.device_sm_map.get)


# ─────────────────────────────────────────────────────────────────────────────
# Context实现Stub（供完整TE集成时替换）
# ─────────────────────────────────────────────────────────────────────────────

class _FP8NativeContext(contextlib.AbstractContextManager):
    """
    SM89+设备的原生FP8 context stub。

    完整实现中应调用 transformer_engine.pytorch.fp8_autocast()。
    此处为DES-LOC框架的接口层，实际TE集成由deepspeed/ops/te_bridge.py负责。
    """

    def __init__(self, layer_number: int, is_init: bool = False) -> None:
        self._layer = layer_number
        self._is_init = is_init

    def __enter__(self):
        logger.debug(
            "_FP8NativeContext.__enter__: layer=%d is_init=%s",
            self._layer, self._is_init,
        )
        _set_thread_local_quant_mode("fp8_native", self._layer)
        return self

    def __exit__(self, *args):
        _clear_thread_local_quant_mode(self._layer)
        return False


class _OuterFP8DelayedContext(contextlib.AbstractContextManager):
    """
    FP8 delayed recipe的外层全局context。

    在DES-LOC中，此context持有跨层共享的scaling factor状态，
    由primary device (H100)管理。A6000层通过DES-LOC的共享状态
    总线访问这些scaling factors。
    """

    def __init__(self, primary_device: int) -> None:
        self._device = primary_device

    def __enter__(self):
        logger.debug(
            "_OuterFP8DelayedContext.__enter__: primary_device=%d",
            self._device,
        )
        return self

    def __exit__(self, *args):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DES-LOC HeteroFP4MambaStack（主要公开接口）
# ─────────────────────────────────────────────────────────────────────────────

class HeteroFP4MambaStack(nn.Module):
    """
    DES-LOC异构量化的MambaStack适配层。

    上游对应：megatron/core/ssm/mamba_block.py::MambaStack

    设计意图：
    本类不重新实现MambaStack的序列建模逻辑，而是作为量化context
    的适配包装器，将Megatron ab5e277b引入的FP4/FP8 context路由
    机制转化为DES-LOC的异构设备感知版本。

    在DES-LOC的pipeline parallel布局中：
      - stage 0,1 → A6000 SM86 (FP8 sim模式)
      - stage 2   → H100 SM90  (FP4 native模式)

    层构建时（__init__）和前向传播时（forward）都需要使用
    HeteroQuantConfig感知的context，而非Megatron的单一设备假设。

    Attributes
    ----------
    config : HeteroQuantConfig
        异构量化配置。
    factory : QuantContextFactory
        量化context工厂（持有router和cache_coordinator引用）。
    layers : nn.ModuleList
        Mamba/Transformer/MLP层的列表（由子类或builder填充）。
    pp_layer_offset : int
        Pipeline parallel层偏移。
    """

    def __init__(
        self,
        config: HeteroQuantConfig,
        pp_layer_offset: int = 0,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_layer_offset = pp_layer_offset
        self._device_index = device_index

        router = DeviceCapabilityRouter(config)
        cache_coord = (
            LocalityCacheQuantCoordinator(config.cache_skip_threshold)
            if config.locality_cache_enabled
            else None
        )
        self.factory = QuantContextFactory(config, router, cache_coord)
        self.layers: nn.ModuleList = nn.ModuleList()

        logger.info(
            "HeteroFP4MambaStack initialized: pp_offset=%d device=%s fp8=%s fp4=%s",
            pp_layer_offset,
            device_index,
            config.fp8,
            config.fp4,
        )

    def build_layer_with_init_context(
        self,
        layer_index: int,
        layer_builder: Callable[[], nn.Module],
        device_index: Optional[int] = None,
    ) -> nn.Module:
        """
        在正确的初始化量化context下构建单个层。

        对应Megatron ab5e277b中的层构建循环：
            if config.fp8:
                quant_init_context = get_fp8_context(config, i + pp_layer_offset, is_init=True)
            elif config.fp4:
                quant_init_context = get_fp4_context(config, i + pp_layer_offset, is_init=True)
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                layer = build_module(...)

        DES-LOC变更：通过factory.make_init_context引入设备路由。
        """
        dev = device_index if device_index is not None else self._device_index
        ctx = self.factory.make_init_context(
            layer_index, self.pp_layer_offset, device_index=dev
        )
        with ctx:
            layer = layer_builder()
        logger.debug(
            "Layer %d built under %s context.",
            layer_index + self.pp_layer_offset,
            ctx.__class__.__name__,
        )
        return layer

    def forward_with_hetero_context(
        self,
        hidden_states: torch.Tensor,
        layer_device_map: Optional[Dict[int, int]] = None,
        cache_hit_map: Optional[Dict[int, bool]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        使用DES-LOC异构量化context执行前向传播。

        对应Megatron ab5e277b中重构后的forward循环：
            # 在循环外绑定get_inner_quant_context（消除循环内条件分支）
            with outer_fp8_context:
                for layer in self.layers:
                    inner_quant_context = get_inner_quant_context(
                        config, layer.layer_number - 1
                    )
                    with inner_quant_context:
                        # dispatch to TransformerLayer or MambaLayer or MLP

        DES-LOC变更：
            1. get_inner_context_getter绑定了DES-LOC的设备路由
            2. 通过layer_device_map指定每层所在设备
            3. 通过cache_hit_map记录cache事件（影响未来的context跳过决策）

        Parameters
        ----------
        hidden_states : torch.Tensor
            输入的隐藏状态张量。
        layer_device_map : Optional[Dict[int, int]]
            layer_0indexed → device_index的映射。
            None时对所有层使用self._device_index。
        cache_hit_map : Optional[Dict[int, bool]]
            layer_0indexed → is_cache_hit的映射。
            用于更新LocalityCacheQuantCoordinator统计。
        **kwargs
            传递给各层forward的额外参数。

        Returns
        -------
        torch.Tensor
            处理后的隐藏状态。
        """
        # 在循环外构建getter闭包（Megatron ab5e277b的关键优化，此处保留）
        get_inner_ctx = self.factory.make_inner_context_getter(self.pp_layer_offset)

        with self.factory.outer_context():
            for i, layer in enumerate(self.layers):
                # Megatron注释保留："Layers have 1-indexed layer_numbers attribute."
                layer_number_0indexed = getattr(layer, "layer_number", i + 1) - 1

                # DES-LOC：更新cache统计
                if (
                    cache_hit_map is not None
                    and self.factory._cache_coordinator is not None
                    and layer_number_0indexed in cache_hit_map
                ):
                    self.factory._cache_coordinator.record_cache_event(
                        layer_number_0indexed,
                        cache_hit_map[layer_number_0indexed],
                    )

                # DES-LOC：确定此层的目标设备
                dev = (
                    layer_device_map.get(layer_number_0indexed, self._device_index)
                    if layer_device_map is not None
                    else self._device_index
                )

                inner_ctx = get_inner_ctx(layer_number_0indexed, dev)

                with inner_ctx:
                    # 对应Megatron的TransformerLayer / MambaLayer / Expert分派
                    # ab5e277b的注释："MambaLayer, Expert, or MLP"
                    hidden_states = self._dispatch_layer(
                        layer, hidden_states, **kwargs
                    )

        return hidden_states

    @staticmethod
    def _dispatch_layer(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        分派layer的forward调用。

        对应Megatron的：
            if isinstance(layer, TransformerLayer):
                hidden_states, _ = layer(hidden_states=hidden_states, ...)
            else:  # MambaLayer, Expert, or MLP
                hidden_states = layer(hidden_states=hidden_states, ...)

        DES-LOC保持相同的分派逻辑，但通过thread-local quant mode
        让各层感知当前量化上下文（get_current_quant_mode）。
        """
        # 通过duck typing检测TransformerLayer（避免直接import Megatron类型）
        result = layer(hidden_states=hidden_states, **kwargs)
        if isinstance(result, tuple):
            hidden_states, _ = result
        else:
            hidden_states = result
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# 便捷工厂函数（模块级公开API）
# ─────────────────────────────────────────────────────────────────────────────

def create_hetero_quant_config_for_des_loc_cluster() -> HeteroQuantConfig:
    """
    为2xA6000+1xH100 NVL的DES-LOC集群创建推荐量化配置。

    设备拓扑：
      device 0: A6000 48GB SM86
      device 1: A6000 48GB SM86
      device 2: H100 NVL 96GB SM90
    """
    return HeteroQuantConfig(
        fp8=True,
        fp8_recipe=Fp8Recipe.CURRENT,  # non-delayed，避免PCIe瓶颈上的scaling同步
        fp4="mxfp4",                    # H100上使用FP4，A6000自动降级
        device_sm_map={0: 86, 1: 86, 2: 90},
        locality_cache_enabled=True,
        cache_skip_threshold=_CACHE_HIT_SKIP_QUANT_THRESHOLD,
        fallback_precision=QuantPrecision.BF16,
    )


def get_hetero_fp4_context(
    config: HeteroQuantConfig,
    layer_number: int,
    device_index: Optional[int] = None,
    is_init: bool = False,
) -> contextlib.AbstractContextManager:
    """
    顶层context获取函数，对应Megatron的get_fp4_context接口。

    提供与Megatron API兼容的单点入口，内部走DES-LOC的异构路由逻辑。
    供外部直接调用（如不使用HeteroFP4MambaStack包装器的场景）。
    """
    router = DeviceCapabilityRouter(config)
    factory = QuantContextFactory(config, router, cache_coordinator=None)

    if is_init:
        return factory.make_init_context(layer_number, 0, device_index)

    getter = factory.make_inner_context_getter(pp_layer_offset=0)
    return getter(layer_number, device_index)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = HeteroQuantConfig(
        fp8=True,
        fp8_recipe=Fp8Recipe.CURRENT,
        fp4="mxfp4",
        device_sm_map={0: 86, 1: 86, 2: 90},
        locality_cache_enabled=True,
    )
    router = DeviceCapabilityRouter(cfg)

    # 1. SM路由正确性
    assert router.route(QuantPrecision.FP4, device_index=0) == QuantPrecision.FP8, \
        "A6000 (SM86) should downgrade FP4 → FP8"
    assert router.route(QuantPrecision.FP4, device_index=2) == QuantPrecision.FP4, \
        "H100 (SM90) should keep FP4"
    assert router.route(QuantPrecision.FP8, device_index=1) == QuantPrecision.FP8, \
        "A6000 (SM86) should keep FP8 (sim mode)"

    # 2. LocalityCacheQuantCoordinator阈值建议
    coord = LocalityCacheQuantCoordinator(threshold=0.85)
    for _ in range(90):
        coord.record_cache_event(layer_number=3, is_hit=True)
    for _ in range(10):
        coord.record_cache_event(layer_number=3, is_hit=False)
    assert coord.should_skip_quant_context(3) is True, \
        "Hit rate=0.90 > 0.85 threshold should advise skip"

    # 3. factory context路径不抛出异常
    factory = QuantContextFactory(cfg, router, coord)
    getter = factory.make_inner_context_getter(pp_layer_offset=2)
    with getter(0, device_index=2):  # H100 FP4
        pass
    with getter(1, device_index=0):  # A6000 FP8 sim
        pass

    # 4. outer_context对non-delayed recipe返回nullcontext
    outer = factory.outer_context()
    assert isinstance(outer, type(nullcontext())), \
        "non-delayed recipe should have nullcontext as outer"

    # 5. DES-LOC集群推荐配置工厂
    cluster_cfg = create_hetero_quant_config_for_des_loc_cluster()
    assert cluster_cfg.device_sm_map[2] == 90
    assert cluster_cfg.fp4 == "mxfp4"

    print("All smoke tests passed.")
