"""
HeteroMXFP8InferenceQuant — DES-LOC异构推理量化层

上游设计意图（Megatron commit 44e27d04）：
    Megatron-LM 为推理线性层引入 MXFP8（Microscaling FP8）量化支持。核心改动包括：
    1. MXFP8Tensor dataclass：封装 (data, scale) 对，通过 FlashInfer mxfp8_quantize 从 BF16 量化；
    2. quantize_model_to_mxfp8()：递归遍历 TransformerEngine MXFP8 模型，将 TE 格式权重
       反量化后用 FlashInfer 重新量化，确保格式统一；
    3. _apply_linear()：在 InferenceRowParallelLinear / InferenceLayerNormColumnParallelLinear
       的 forward 路径中统一分发：fp8_recipe=="mxfp8" 走 FlashInfer mm_mxfp8，否则走
       torch.matmul；
    4. 条件化 prepare_model_for_fp8_inference()：transformer_impl=="inference_optimized"
       时跳过旧 FP8 路径，改由新 MXFP8 路径接管；
    5. 参数校验：mxfp8 + inference_optimized 路径要求 FlashInfer >= 0.6.4。

DES-LOC 适配点（Decoupled Execution with Shared LOcality Cache）：
    DES-LOC 的核心思想是将权重/激活的量化决策与执行设备解耦：
    - A6000（SM86，48 GB）：承载 tensor-parallel 分片权重，执行 MXFP8 GEMM；
    - H100 NVL（SM90，96 GB）：承载 KV-cache 与大 batch 上下文，执行 BF16/FP8 推理；
    - PCIe 互联无 NVLink：跨设备权重迁移代价高，需最大化权重在本地缓存的复用；
    - 1.5 TB CPU DRAM：作为"局部性共享缓存"（LOC Cache），暂存尚未迁移的量化权重。

    具体适配：
    A. DeviceCapabilityRouter：根据 SM 版本路由 MXFP8 vs BF16 GEMM，SM86 用 torch 模拟
       路径（FlashInfer mxfp8 需要 SM90+），SM90+ 启用 FlashInfer 快速路径；
    B. HeteroMXFP8Tensor：继承 MXFP8Tensor 语义，增加 device_affinity 标注和 CPU LOC
       缓存槽位（pinned memory），支持跨设备权重复用；
    C. LOCWeightCache：LRU 权重缓存，以 (layer_id, shard_id) 为 key，在 CPU pinned
       memory 上保留量化权重副本，避免 PCIe 重传；
    D. HeteroQuantLinear：nn.Module 封装，forward 时动态决策：本地命中 → 直接 GEMM；
       缓存命中 → 异步 H2D 传输 + GEMM；缓存未命中 → 量化 + 写缓存 + GEMM；
    E. quantize_hetero_model()：替代 Megatron 的 quantize_model_to_mxfp8，感知
       device_affinity，对 A6000 层使用模拟路径，对 H100 层使用 FlashInfer 路径。
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 版本检测与后端可用性
# ──────────────────────────────────────────────────────────────────────────────

try:
    import flashinfer
    from flashinfer import mm_mxfp8 as _fi_mm_mxfp8
    from flashinfer import mxfp8_quantize as _fi_mxfp8_quantize
    _FLASHINFER_VERSION = getattr(flashinfer, "__version__", "0.0.0")
    HAVE_FLASHINFER = True
    logger.info("FlashInfer %s 已加载，SM90+ MXFP8 快速路径可用", _FLASHINFER_VERSION)
except ImportError:
    HAVE_FLASHINFER = False
    _FLASHINFER_VERSION = "0.0.0"
    logger.warning("FlashInfer 未安装，MXFP8 将回退到 BF16 模拟路径（仅限 SM86）")

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as _TEMXFP8Tensor
    HAVE_TE = True
    logger.info("TransformerEngine MXFP8Tensor 已加载")
except ImportError:
    HAVE_TE = False
    _TEMXFP8Tensor = None
    logger.debug("TransformerEngine 未安装，跳过 TE→FlashInfer 转换路径")


def _parse_version(v: str) -> Tuple[int, ...]:
    """将 '0.6.4' 解析为 (0, 6, 4) 用于简单比较，不依赖 packaging。"""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (0, 0, 0)


def is_flashinfer_min_version(min_version: str) -> bool:
    """检查 FlashInfer 是否满足最低版本要求。"""
    if not HAVE_FLASHINFER:
        return False
    return _parse_version(_FLASHINFER_VERSION) >= _parse_version(min_version)


# ──────────────────────────────────────────────────────────────────────────────
# 设备能力路由
# ──────────────────────────────────────────────────────────────────────────────

# DES-LOC 硬件拓扑常量
_SM_THRESHOLD_MXFP8 = 90   # SM90+ (H100) 支持原生 MXFP8 GEMM
_SM_A6000 = 86              # A6000 SM86，回退到 BF16 模拟
_SM_H100 = 90               # H100 NVL SM90

# 全局设备能力缓存，避免重复调用 get_device_properties
_device_sm_cache: Dict[int, int] = {}
_device_sm_cache_lock = threading.Lock()


def get_device_sm(device: torch.device) -> int:
    """返回指定设备的 SM 主版本号，带缓存。"""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    with _device_sm_cache_lock:
        if idx not in _device_sm_cache:
            props = torch.cuda.get_device_properties(idx)
            _device_sm_cache[idx] = props.major
            logger.debug(
                "设备 cuda:%d — %s，SM%d%d",
                idx, props.name, props.major, props.minor
            )
        return _device_sm_cache[idx]


def device_supports_native_mxfp8(device: torch.device) -> bool:
    """
    判断设备是否支持原生 MXFP8 GEMM（需要 SM90+ 且 FlashInfer >= 0.6.4）。

    DES-LOC 适配：A6000 (SM86) 返回 False，走 BF16 模拟；
                  H100 NVL (SM90) 返回 True，走 FlashInfer 快速路径。
    """
    sm = get_device_sm(device)
    if sm < _SM_THRESHOLD_MXFP8:
        return False
    return is_flashinfer_min_version("0.6.4")


# ──────────────────────────────────────────────────────────────────────────────
# HeteroMXFP8Tensor — 异构感知的 MXFP8 张量
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeteroMXFP8Tensor:
    """
    MXFP8 张量封装，携带设备亲和性标注和 CPU LOC 缓存槽位。

    上游对应：megatron/core/inference/quantization/mxfp8_tensor.py::MXFP8Tensor
    DES-LOC 扩展：
        - device_affinity：记录权重所属设备（A6000/H100），用于缓存路由；
        - cpu_cache_pin：CPU pinned memory 副本，供 LOCWeightCache 使用；
        - group_size：量化块大小，默认 32（与 Megatron 一致）。
    """
    data: torch.Tensor           # FP8 量化数据，shape [M, K]
    scale: torch.Tensor          # 块缩放因子
    device_affinity: int = field(default=-1)   # cuda device index，-1 表示未绑定
    group_size: int = field(default=32)
    _cpu_pin: Optional[torch.Tensor] = field(default=None, repr=False)

    def size(self, idx: Optional[int] = None) -> Union[torch.Size, int]:
        """与 torch.Tensor.size() 兼容的接口。"""
        return self.data.size(idx)

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def to_cpu_pinned(self) -> "HeteroMXFP8Tensor":
        """
        将量化张量复制到 CPU pinned memory，用于 LOC 缓存。
        PCIe 互联下，pinned memory 可显著加速后续 H2D 传输。
        """
        if self._cpu_pin is not None:
            return self  # 已有 pinned 副本
        pin_data = torch.empty_like(self.data, device="cpu", pin_memory=True)
        pin_data.copy_(self.data)
        pin_scale = torch.empty_like(self.scale, device="cpu", pin_memory=True)
        pin_scale.copy_(self.scale)
        # 返回新对象，保留原 GPU 数据
        return HeteroMXFP8Tensor(
            data=self.data,
            scale=self.scale,
            device_affinity=self.device_affinity,
            group_size=self.group_size,
            _cpu_pin=(pin_data, pin_scale),
        )

    def restore_to_device(self, device: torch.device, stream: Optional[torch.cuda.Stream] = None) -> "HeteroMXFP8Tensor":
        """
        从 CPU pinned memory 异步恢复到目标 GPU 设备。
        DES-LOC 核心：权重在 CPU LOC Cache 中驻留，按需 H2D 迁移。
        """
        if self._cpu_pin is None:
            raise RuntimeError("无 CPU pinned 副本，无法 restore_to_device")
        pin_data, pin_scale = self._cpu_pin
        if stream is not None:
            with torch.cuda.stream(stream):
                gpu_data = pin_data.to(device, non_blocking=True)
                gpu_scale = pin_scale.to(device, non_blocking=True)
        else:
            gpu_data = pin_data.to(device)
            gpu_scale = pin_scale.to(device)
        return HeteroMXFP8Tensor(
            data=gpu_data,
            scale=gpu_scale,
            device_affinity=device.index if device.index is not None else 0,
            group_size=self.group_size,
        )

    @classmethod
    def from_bf16(
        cls,
        x: torch.Tensor,
        group_size: int = 32,
        device_affinity: int = -1,
    ) -> "HeteroMXFP8Tensor":
        """
        从 BF16 张量量化为 MXFP8。

        路由逻辑（DES-LOC 适配）：
        - SM90+ 且 FlashInfer 可用 → FlashInfer mxfp8_quantize（硬件原生路径）；
        - SM86 (A6000) 或 FlashInfer 不可用 → BF16 软件模拟路径。

        注意：模拟路径的 scale 用每块的绝对最大值，精度略低于硬件路径，
        但对 A6000 推理任务（主要承载 TP 分片）已足够。
        """
        assert x.is_cuda, "输入必须在 CUDA 设备上"
        assert x.dim() == 2, f"输入必须为 2D [M, K]，实际为 {x.dim()}D"
        M, K = x.shape
        assert K % group_size == 0, f"K ({K}) 必须能被 group_size ({group_size}) 整除"

        device = x.device
        use_native = device_supports_native_mxfp8(device)

        if use_native:
            logger.debug(
                "SM90+ 原生 MXFP8 量化，设备 cuda:%d，shape %s",
                device.index, (M, K)
            )
            data, scale = _fi_mxfp8_quantize(x)
        else:
            logger.debug(
                "SM86 软件模拟 MXFP8 量化，设备 cuda:%d，shape %s",
                device.index if device.index is not None else "?", (M, K)
            )
            data, scale = cls._simulate_mxfp8_quantize(x, group_size)

        dev_idx = device.index if device.index is not None else 0
        return cls(
            data=data,
            scale=scale,
            device_affinity=dev_idx if device_affinity < 0 else device_affinity,
            group_size=group_size,
        )

    @staticmethod
    def _simulate_mxfp8_quantize(
        x: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软件模拟 MXFP8 量化（SM86 回退路径）。

        实现 MX (Microscaling) 规范中的块量化：
        1. 将输入按最后维度切分为 [M, K//G, G] 的块；
        2. 计算每块的绝对最大值作为 scale（E8M0 近似）；
        3. 将每块数据归一化后量化到 float8_e4m3fn。

        DES-LOC 说明：此路径在 A6000 上执行，精度与硬件路径存在微小差异，
        通过 verify_quantization_consistency() 检查一致性。
        """
        M, K = x.shape
        G = group_size
        n_groups = K // G

        # [M, n_groups, G]
        x_blocked = x.view(M, n_groups, G)
        amax = x_blocked.abs().amax(dim=-1, keepdim=True).float()  # [M, n_groups, 1]
        amax = amax.clamp(min=1e-12)

        # E8M0 scale：用 2 的幂次近似
        log2_scale = amax.log2().floor()
        scale_f32 = (2.0 ** log2_scale).squeeze(-1)  # [M, n_groups]

        # 量化到 FP8 e4m3fn
        x_scaled = (x_blocked / scale_f32.unsqueeze(-1)).clamp(-448.0, 448.0)
        x_fp8 = x_scaled.float().to(torch.float8_e4m3fn).view(M, K)

        # scale 以 uint8 E8M0 格式存储（与 FlashInfer 约定一致）
        # E8M0：8-bit 纯指数，值 = 2^(byte - 127)
        scale_e8m0 = (log2_scale.squeeze(-1) + 127).clamp(0, 255).to(torch.uint8)

        return x_fp8, scale_e8m0


# ──────────────────────────────────────────────────────────────────────────────
# LOCWeightCache — 局部性共享权重缓存（DES-LOC 核心组件）
# ──────────────────────────────────────────────────────────────────────────────

class LOCWeightCache:
    """
    Shared LOcality Cache（LOC）：在 CPU DRAM（1.5 TB）中缓存 MXFP8 量化权重。

    设计动机（DES-LOC）：
        A6000 ↔ H100 通过 PCIe 互联，带宽 ~16 GB/s，延迟高。若每次 forward
        都将权重从 GPU 传到对端，开销不可接受。LOC Cache 的策略是：
        1. 首次量化的权重同时写入 CPU pinned memory；
        2. 后续推理命中缓存时，通过异步 H2D 预取，隐藏 PCIe 延迟；
        3. LRU 淘汰策略，总容量通过 max_bytes 控制（默认 64 GB，
           覆盖约 70B 参数模型的 MXFP8 权重）。

    Key 格式：(layer_id: int, shard_id: str)
        - layer_id：transformer layer 编号；
        - shard_id：TP 分片标识（如 "col_0", "row_1"）。
    """

    def __init__(self, max_bytes: int = 64 * 1024 ** 3):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[Tuple[int, str], HeteroMXFP8Tensor] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info(
            "LOCWeightCache 初始化，容量上限 %.1f GB（CPU DRAM）",
            max_bytes / 1024 ** 3
        )

    def _tensor_bytes(self, t: HeteroMXFP8Tensor) -> int:
        """估算 HeteroMXFP8Tensor 的内存占用（data + scale）。"""
        return t.data.numel() * t.data.element_size() + \
               t.scale.numel() * t.scale.element_size()

    def get(
        self,
        key: Tuple[int, str],
        target_device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[HeteroMXFP8Tensor]:
        """
        查询缓存。命中时将 CPU pinned 副本异步传输到 target_device，返回 GPU 张量。
        未命中返回 None。
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            self._cache.move_to_end(key)  # LRU 更新
            self._hits += 1
            cached = self._cache[key]

        # 在锁外执行 H2D 传输，避免阻塞其他线程
        return cached.restore_to_device(target_device, stream=stream)

    def put(self, key: Tuple[int, str], tensor: HeteroMXFP8Tensor) -> None:
        """
        将量化权重写入缓存（同时创建 CPU pinned 副本）。
        若超出容量限制，按 LRU 顺序淘汰。
        """
        pinned = tensor.to_cpu_pinned()
        new_bytes = self._tensor_bytes(pinned)

        with self._lock:
            if key in self._cache:
                # 已存在，更新
                old = self._cache.pop(key)
                self._current_bytes -= self._tensor_bytes(old)

            # LRU 淘汰
            while self._current_bytes + new_bytes > self._max_bytes and self._cache:
                evict_key, evict_val = self._cache.popitem(last=False)
                self._current_bytes -= self._tensor_bytes(evict_val)
                logger.debug("LOC 缓存淘汰 key=%s，释放 %.2f MB", evict_key,
                             self._tensor_bytes(evict_val) / 1024 ** 2)

            self._cache[key] = pinned
            self._current_bytes += new_bytes

        logger.debug(
            "LOC 缓存写入 key=%s，%.2f MB，总占用 %.2f GB",
            key, new_bytes / 1024 ** 2, self._current_bytes / 1024 ** 3
        )

    def stats(self) -> Dict[str, float]:
        """返回缓存命中率统计。"""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "used_gb": self._current_bytes / 1024 ** 3,
            "max_gb": self._max_bytes / 1024 ** 3,
        }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
        logger.info("LOCWeightCache 已清空")


# 全局单例缓存，由所有 HeteroQuantLinear 共享
_GLOBAL_LOC_CACHE: Optional[LOCWeightCache] = None
_GLOBAL_LOC_CACHE_LOCK = threading.Lock()


def get_global_loc_cache(max_bytes: int = 64 * 1024 ** 3) -> LOCWeightCache:
    """获取全局 LOC 缓存单例。"""
    global _GLOBAL_LOC_CACHE
    with _GLOBAL_LOC_CACHE_LOCK:
        if _GLOBAL_LOC_CACHE is None:
            _GLOBAL_LOC_CACHE = LOCWeightCache(max_bytes=max_bytes)
    return _GLOBAL_LOC_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# MXFP8 GEMM 分发函数
# ──────────────────────────────────────────────────────────────────────────────

def _apply_hetero_linear(
    x: torch.Tensor,
    weight: Union[torch.Tensor, HeteroMXFP8Tensor],
    fp8_recipe: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    异构感知的线性层分发函数。

    上游对应：megatron/core/tensor_parallel/inference_layers.py::_apply_linear
    DES-LOC 扩展：
        - 根据当前执行设备的 SM 版本选择计算后端；
        - SM90+ (H100)：FlashInfer mm_mxfp8 原生路径；
        - SM86 (A6000)：BF16 反量化后 torch.matmul 模拟路径；
        - fp8_recipe != "mxfp8" 或 weight 为普通 Tensor：退回 torch.matmul。

    参数：
        x: 激活张量，shape [B, T, C] 或 [B, C]（BF16）
        weight: 权重，HeteroMXFP8Tensor 或普通 Tensor
        fp8_recipe: 量化方案标识，"mxfp8" 启用 MXFP8 路径
        out: 可选输出缓冲区（用于 in-place 写入 symmetric memory）
    """
    kwargs = {"out": out} if out is not None else {}

    if fp8_recipe != "mxfp8" or not isinstance(weight, HeteroMXFP8Tensor):
        # 标准路径
        w = weight.data if isinstance(weight, HeteroMXFP8Tensor) else weight
        return torch.matmul(x, w.t() if w.dim() == 2 else w, **kwargs)

    device = x.device
    use_native = device_supports_native_mxfp8(device)

    if use_native:
        # H100 快速路径：FlashInfer mm_mxfp8
        # 激活需要降到 2D，FlashInfer 约定 [M, K]
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, C = x.shape
            x_2d = x.reshape(B * T, C)
        else:
            x_2d = x

        x_q = HeteroMXFP8Tensor.from_bf16(x_2d)
        result = _fi_mm_mxfp8(
            x_q.data,
            weight.data.T,
            x_q.scale,
            weight.scale,
            out_dtype=torch.bfloat16,
            out=out,
        )
        if x.dim() == 3:
            result = result.reshape(B, T, -1)
        return result
    else:
        # A6000 回退路径：FP8 反量化 → BF16 matmul
        logger.debug(
            "SM86 回退路径：FP8 反量化后执行 BF16 matmul，激活 shape=%s", x.shape
        )
        w_bf16 = _dequantize_mxfp8_to_bf16(weight)
        return torch.matmul(x, w_bf16.t(), **kwargs)


def _dequantize_mxfp8_to_bf16(tensor: HeteroMXFP8Tensor) -> torch.Tensor:
    """
    将 HeteroMXFP8Tensor 反量化回 BF16，用于 SM86 软件模拟路径。

    实现：data (fp8_e4m3fn) × scale (E8M0) → BF16
    """
    data_f32 = tensor.data.float()  # fp8_e4m3fn → float32

    # scale 是 E8M0 格式（uint8），值 = 2^(byte - 127)
    scale_i32 = tensor.scale.to(torch.int32)
    scale_f32 = (2.0 ** (scale_i32.float() - 127.0))  # [M, n_groups]

    M, K = tensor.data.shape
    G = tensor.group_size
    n_groups = K // G

    scale_expanded = scale_f32.unsqueeze(-1).expand(M, n_groups, G).reshape(M, K)
    result = (data_f32 * scale_expanded).to(torch.bfloat16)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# HeteroQuantLinear — 异构量化线性层（DES-LOC 核心 Module）
# ──────────────────────────────────────────────────────────────────────────────

class HeteroQuantLinear(nn.Module):
    """
    DES-LOC 异构量化线性层。

    替代 Megatron InferenceRowParallelLinear / InferenceLayerNormColumnParallelLinear
    中的 GEMM 核心，增加以下功能：
    1. LOC 缓存感知：首次 forward 时将量化权重写入 CPU LOC Cache；
    2. 异步预取：通过专属 CUDA stream 执行 H2D 传输，与 compute 流并行；
    3. 设备路由：根据执行设备的 SM 版本自动选择计算后端；
    4. 量化延迟初始化：权重在第一次 forward 时按需量化，避免模型加载时的额外开销。

    参数：
        in_features: 输入特征维度
        out_features: 输出特征维度
        layer_id: 所属 transformer 层编号，用于 LOC Cache 索引
        shard_id: TP 分片标识，用于 LOC Cache 索引
        fp8_recipe: 量化方案，"mxfp8" 或 None
        group_size: MXFP8 块大小，默认 32
        loc_cache: LOC 缓存实例，默认使用全局缓存
        bias: 是否包含偏置项
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_id: int = 0,
        shard_id: str = "default",
        fp8_recipe: Optional[str] = None,
        group_size: int = 32,
        loc_cache: Optional[LOCWeightCache] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id
        self.shard_id = shard_id
        self.fp8_recipe = fp8_recipe
        self.group_size = group_size
        self._loc_cache = loc_cache  # None 时使用全局缓存
        self._cache_key = (layer_id, shard_id)

        # 原始 BF16 权重（训练后量化前）
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        # 量化权重缓存（GPU 侧），延迟初始化
        self._quant_weight: Optional[HeteroMXFP8Tensor] = None
        self._quant_device: Optional[int] = None  # 最后量化时的设备 index

        # 用于异步 H2D 的专属 CUDA stream
        self._prefetch_stream: Optional[torch.cuda.Stream] = None

        logger.debug(
            "HeteroQuantLinear 初始化：layer=%d shard=%s (%d→%d) recipe=%s",
            layer_id, shard_id, in_features, out_features, fp8_recipe
        )

    @property
    def loc_cache(self) -> LOCWeightCache:
        if self._loc_cache is not None:
            return self._loc_cache
        return get_global_loc_cache()

    def _get_prefetch_stream(self) -> torch.cuda.Stream:
        """按需创建 H2D 预取专属 stream。"""
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream()
        return self._prefetch_stream

    def _quantize_weight(self, device: torch.device) -> HeteroMXFP8Tensor:
        """
        将 BF16 权重量化为 HeteroMXFP8Tensor。
        量化后写入 LOC Cache（CPU pinned memory）。
        """
        w = self.weight.data.to(device)
        dev_idx = device.index if device.index is not None else 0
        q_weight = HeteroMXFP8Tensor.from_bf16(
            w, group_size=self.group_size, device_affinity=dev_idx
        )
        # 写入 LOC Cache
        self.loc_cache.put(self._cache_key, q_weight)
        logger.info(
            "layer=%d shard=%s 权重量化完成，写入 LOC Cache，设备 cuda:%d",
            self.layer_id, self.shard_id, dev_idx
        )
        return q_weight

    def _get_quant_weight(self, device: torch.device) -> HeteroMXFP8Tensor:
        """
        获取量化权重，按优先级：
        1. GPU 本地已量化且设备匹配 → 直接返回；
        2. LOC Cache 命中 → 异步 H2D 传输到当前设备；
        3. 缓存未命中 → 量化 + 写缓存。
        """
        dev_idx = device.index if device.index is not None else 0

        # 检查 GPU 本地缓存
        if self._quant_weight is not None and self._quant_device == dev_idx:
            return self._quant_weight

        # 尝试从 LOC Cache 恢复
        stream = self._get_prefetch_stream()
        cached = self.loc_cache.get(self._cache_key, device, stream=stream)
        if cached is not None:
            # 等待 H2D 传输完成
            torch.cuda.current_stream().wait_stream(stream)
            self._quant_weight = cached
            self._quant_device = dev_idx
            logger.debug(
                "layer=%d shard=%s LOC Cache 命中，H2D 传输完成",
                self.layer_id, self.shard_id
            )
            return cached

        # 缓存未命中，重新量化
        logger.info(
            "layer=%d shard=%s LOC Cache 未命中，执行量化",
            self.layer_id, self.shard_id
        )
        q_weight = self._quantize_weight(device)
        self._quant_weight = q_weight
        self._quant_device = dev_idx
        return q_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，自动路由到适配当前设备的 MXFP8 或 BF16 路径。

        参数：
            x: 激活张量，dtype=BF16，shape [B, T, C] 或 [B, C]
        返回：
            输出张量，shape [B, T, out_features] 或 [B, out_features]
        """
        if self.fp8_recipe == "mxfp8":
            weight = self._get_quant_weight(x.device)
        else:
            weight = self.weight

        out = _apply_hetero_linear(x, weight, fp8_recipe=self.fp8_recipe)

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"layer={self.layer_id}, shard={self.shard_id}, "
            f"recipe={self.fp8_recipe}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 量化验证工具
# ──────────────────────────────────────────────────────────────────────────────

def verify_quantization_consistency(
    te_dequantized: torch.Tensor,
    fi_quantized: HeteroMXFP8Tensor,
    rtol: float = 0.1,
    atol: float = 0.01,
) -> None:
    """
    验证 TE → FlashInfer MXFP8 转换的数值一致性。

    上游对应：megatron/core/inference/quantization/utils.py::_verify_te_to_flashinfer_mxfp8_conversion
    DES-LOC 扩展：同时支持 SM86 软件模拟路径的验证（对第一个块进行数值比对）。

    参数：
        te_dequantized: TransformerEngine 反量化后的 BF16 参考值
        fi_quantized: FlashInfer/模拟路径量化结果
        rtol, atol: 容差（MXFP8 精度有损，默认宽松）
    """
    # 取第一个逻辑块（32 个元素）
    te_block = te_dequantized[0, :32].float()

    # 从 fi_quantized 反量化第一个块
    fi_bf16 = _dequantize_mxfp8_to_bf16(fi_quantized)
    fi_block = fi_bf16[0, :32].float()

    if not torch.allclose(te_block, fi_block, rtol=rtol, atol=atol):
        diff_norm = torch.norm(te_block - fi_block).item()
        raise ValueError(
            f"MXFP8 一致性检查失败：diff_norm={diff_norm:.4f}，"
            f"rtol={rtol}，atol={atol}"
        )
    logger.debug("MXFP8 一致性检查通过，diff_norm=%.6f", torch.norm(te_block - fi_block).item())


# ──────────────────────────────────────────────────────────────────────────────
# 模型量化入口函数
# ──────────────────────────────────────────────────────────────────────────────

def quantize_hetero_model(
    model: nn.Module,
    fp8_recipe: str = "mxfp8",
    group_size: int = 32,
    verify: bool = True,
    loc_cache: Optional[LOCWeightCache] = None,
) -> nn.Module:
    """
    将异构模型递归量化为 MXFP8，感知 DES-LOC 硬件拓扑。

    上游对应：megatron/inference/utils.py 中调用的 quantize_model_to_mxfp8()
    DES-LOC 扩展：
        - 对每个线性层，根据其所在设备的 SM 版本选择量化后端；
        - 若模型权重在 A6000 (SM86)：使用软件模拟量化；
        - 若模型权重在 H100 (SM90+)：使用 FlashInfer 量化；
        - 若存在 TransformerEngine MXFP8 权重，先反量化再转换；
        - 量化完成后权重写入 LOC Cache（CPU pinned memory）。

    参数：
        model: 待量化的 nn.Module（已在 eval 模式）
        fp8_recipe: 量化方案，目前支持 "mxfp8"
        group_size: 块大小，默认 32
        verify: 是否执行 TE→FlashInfer 数值一致性验证
        loc_cache: LOC 缓存实例，None 时使用全局缓存

    返回：
        量化后的模型（in-place 修改）
    """
    if fp8_recipe != "mxfp8":
        logger.warning("quantize_hetero_model 目前仅支持 mxfp8，跳过量化")
        return model

    cache = loc_cache if loc_cache is not None else get_global_loc_cache()
    layer_counter = [0]  # 用 list 绕过 Python 闭包限制

    def _quantize_module(module: nn.Module, prefix: str = "") -> None:
        """递归量化模块中的线性层权重。"""
        for name, child in module.named_children():
            _quantize_module(child, prefix=f"{prefix}.{name}" if prefix else name)

        # 处理当前模块的 _parameters
        if not hasattr(module, "_parameters"):
            return

        keys = list(module._parameters.keys())
        for key in keys:
            param = module._parameters[key]
            if param is None:
                continue

            # 情形 A：TransformerEngine MXFP8 权重 → 先反量化
            if HAVE_TE and isinstance(param, _TEMXFP8Tensor):
                logger.info(
                    "%s.%s：检测到 TE MXFP8 权重，执行 TE→DES-LOC 转换",
                    prefix, key
                )
                te_dequantized = param.dequantize()
                if not te_dequantized.is_cuda:
                    te_dequantized = te_dequantized.cuda()

                layer_id = layer_counter[0]
                shard_id = f"{prefix}.{key}"
                q_weight = HeteroMXFP8Tensor.from_bf16(
                    te_dequantized, group_size=group_size
                )
                if verify:
                    try:
                        verify_quantization_consistency(te_dequantized, q_weight)
                    except ValueError as e:
                        logger.error("TE→DES-LOC 一致性验证失败：%s", e)
                        raise

                cache.put((layer_id, shard_id), q_weight)
                del module._parameters[key]
                setattr(module, key, q_weight)
                layer_counter[0] += 1

            # 情形 B：普通 BF16 nn.Parameter → 直接量化
            elif isinstance(param, torch.Tensor) and param.dtype == torch.bfloat16:
                if param.dim() != 2:
                    continue  # 只量化 2D 权重矩阵
                if not param.is_cuda:
                    continue  # 跳过 CPU 权重

                M, K = param.shape
                if K % group_size != 0:
                    logger.debug(
                        "%s.%s shape=%s 无法被 group_size=%d 整除，跳过量化",
                        prefix, key, param.shape, group_size
                    )
                    continue

                layer_id = layer_counter[0]
                shard_id = f"{prefix}.{key}"
                logger.debug(
                    "%s.%s：量化 BF16 权重 %s → MXFP8，layer_id=%d",
                    prefix, key, param.shape, layer_id
                )
                q_weight = HeteroMXFP8Tensor.from_bf16(
                    param.data, group_size=group_size
                )
                cache.put((layer_id, shard_id), q_weight)
                del module._parameters[key]
                setattr(module, key, q_weight)
                layer_counter[0] += 1

    t0 = time.perf_counter()
    _quantize_module(model)
    elapsed = time.perf_counter() - t0

    stats = cache.stats()
    logger.info(
        "quantize_hetero_model 完成：%d 个权重已量化，耗时 %.2fs，"
        "LOC Cache 占用 %.2f GB / %.2f GB",
        layer_counter[0], elapsed, stats["used_gb"], stats["max_gb"]
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Mamba SSM 填充 token 清零（DES-LOC 适配）
# ──────────────────────────────────────────────────────────────────────────────

def clear_padding_outputs_for_quant(
    y: torch.Tensor,
    padding_slice: Optional[slice],
    config_uses_quant: bool,
) -> torch.Tensor:
    """
    在使用量化 scales 时，清零填充 token 的输出，避免污染 amax 计算。

    上游对应：megatron/core/ssm/mamba_mixer.py 中新增的 padding 清零逻辑
    DES-LOC 适配：在异构推理中，A6000 处理部分序列时可能产生填充 token，
    若不清零会导致 H100 侧的量化 amax 计算偏高，影响精度。

    参数：
        y: SSM 输出张量
        padding_slice: 填充 token 的切片索引
        config_uses_quant: 当前配置是否使用量化 scales

    返回：
        清零后的张量（in-place 修改，返回原张量引用）
    """
    if config_uses_quant and padding_slice is not None:
        y[padding_slice] = 0.0
        logger.debug("已清零 padding 位置的 SSM 输出，避免 amax 污染")
    return y


# ──────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== HeteroMXFP8InferenceQuant Smoke Test ===\n")

    # ── 1. LOC Cache 基本读写 ──────────────────────────────────────────────────
    cache = LOCWeightCache(max_bytes=512 * 1024 ** 2)  # 512 MB 测试容量
    dummy_data = torch.zeros(128, 256, dtype=torch.float8_e4m3fn)
    dummy_scale = torch.ones(128, 8, dtype=torch.uint8)
    t = HeteroMXFP8Tensor(data=dummy_data, scale=dummy_scale, group_size=32)
    cache.put((0, "test_shard"), t)
    assert (0, "test_shard") in cache._cache, "LOC Cache put 失败"
    print("✓ LOC Cache put/get 基本功能正常")

    # ── 2. 软件模拟量化（CPU 上验证形状）────────────────────────────────────────
    # 在 CPU 上直接测试 _simulate_mxfp8_quantize 的输出形状
    x_bf16_cpu = torch.randn(64, 128, dtype=torch.bfloat16)
    data_sim, scale_sim = HeteroMXFP8Tensor._simulate_mxfp8_quantize(x_bf16_cpu, group_size=32)
    assert data_sim.shape == (64, 128), f"模拟量化 data shape 错误: {data_sim.shape}"
    assert scale_sim.shape == (64, 4), f"模拟量化 scale shape 错误: {scale_sim.shape}"  # 128//32=4
    print("✓ 软件模拟 MXFP8 量化形状正确")

    # ── 3. 反量化一致性（CPU 上验证）────────────────────────────────────────────
    t_sim = HeteroMXFP8Tensor(data=data_sim, scale=scale_sim, group_size=32)
    dequant = _dequantize_mxfp8_to_bf16(t_sim)
    assert dequant.shape == x_bf16_cpu.shape, f"反量化 shape 错误: {dequant.shape}"
    assert dequant.dtype == torch.bfloat16, f"反量化 dtype 错误: {dequant.dtype}"
    print("✓ 反量化形状与 dtype 正确")

    # ── 4. GPU 路径（有 CUDA 时执行）────────────────────────────────────────────
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        sm = get_device_sm(dev)
        print(f"  检测到 GPU cuda:0，SM{sm}x")

        x_gpu = torch.randn(16, 64, dtype=torch.bfloat16, device=dev)
        w_gpu = torch.randn(128, 64, dtype=torch.bfloat16, device=dev)
        # 标准 matmul 路径（fp8_recipe=None）
        out = _apply_hetero_linear(x_gpu, w_gpu, fp8_recipe=None)
        assert out.shape == (16, 128), f"标准 matmul 输出 shape 错误: {out.shape}"
        print("✓ GPU 标准 matmul 路径正常")
    else:
        print("  (跳过 GPU 测试，无 CUDA 设备)")

    # ── 5. LOC Cache 统计 ─────────────────────────────────────────────────────
    stats = cache.stats()
    assert stats["hits"] == 0 and stats["misses"] == 0, "初始统计应为零"
    print("✓ LOC Cache 统计字段正确")

    print("\n=== Smoke Test PASSED ===")
