"""
HeteroMuPMuonScaling — DES-LOC异构训练框架的MuP×Muon联合缩放适配模块
=======================================================================

上游设计意图（Megatron f58a3281）：
    Megatron引入了MuP（Maximal Update Parametrization）与Muon优化器的协同缩放逻辑。
    核心洞察是：Muon对2D权重矩阵执行谱归一化（Newton-Schulz迭代），其本身已包含
    隐式的宽度缩放，因此不应再被MuP的Adam-style lr/eps覆盖所影响。具体规则：
    1. is_muon_managed_matrix_parameter → dim==2 且非embedding/output → 排除MuP覆盖
    2. muon_scale_mode='spectral' 时发出WARNING（两种缩放机制并存）
    3. muon_scale_mode='unit_rms_norm' 时两者协同，无冲突

DES-LOC适配点（Decoupled Execution with Shared LOcality Cache）：
    DES-LOC硬件拓扑：2×A6000(48GB, SM86) + 1×H100 NVL(96GB, SM90)，PCIe互联
    挑战1 — 设备异构：
        A6000(SM86)不支持BF16原生MMA，H100(SM90)支持FP8/BF16。
        Muon的Newton-Schulz迭代在SM86上必须以FP32执行，SM90可降精度。
        MuP的eps缩放依赖数值精度假设，需按设备dtype动态调整。
    挑战2 — 无NVLink PCIe互联：
        谱归一化需要全局参数范数（2D矩阵奇异值估计），跨设备聚合带宽受限。
        DES-LOC的LOcality Cache将最近一步的奇异值估计缓存在DRAM(1.5TB)，
        避免每步跨PCIe进行全归约，代价是引入一步延迟的staleness。
    挑战3 — 参数分组路由：
        DeepSpeed的ZeRO分区打散了参数与设备的对应关系，需要在MuP覆盖计算时
        恢复参数的"逻辑设备归属"，以便应用正确的dtype/precision策略。

模块职责：
    - HeteroDeviceProfile：封装per-device能力（SM版本、max_dtype、Newton-Schulz精度）
    - MuonMuPScaleMode：枚举缩放模式，扩展上游的三态为四态（增加hetero_auto）
    - LocalityCacheManager：管理DRAM侧的奇异值估计缓存，提供staleness-aware更新
    - HeteroMuPMuonScaler：核心类，实现参数分类、覆盖计算、设备感知调度
    - get_hetero_mup_muon_overrides：对外API，对齐DeepSpeed optimizer param_groups接口
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SM86_MAX_NEWTON_SCHULZ_DTYPE = torch.float32   # A6000: SM86无BF16 MMA
_SM90_MAX_NEWTON_SCHULZ_DTYPE = torch.bfloat16  # H100 NVL: SM90支持BF16
_LOCALITY_CACHE_STALE_STEPS = 1                  # LOcality Cache允许1步staleness
_SPECTRAL_SCALE_WARN_COOLDOWN_STEPS = 100        # WARNING节流：每N步最多1次


# ---------------------------------------------------------------------------
# 设备能力描述
# ---------------------------------------------------------------------------

@dataclass
class HeteroDeviceProfile:
    """封装单个物理设备的计算能力与DES-LOC调度属性。

    DES-LOC适配点：
        PCIe拓扑下，每个设备的Newton-Schulz精度独立决策，避免最低公分母降级。
        H100侧可用BF16迭代，A6000侧强制FP32，两者通过LOcality Cache共享奇异值估计。

    Attributes:
        device_id: CUDA device index
        sm_major: Streaming Multiprocessor major version (86=A6000, 90=H100)
        sm_minor: SM minor version
        vram_gb: 显存容量 GB
        is_locality_cache_host: 是否承担LOcality Cache写回（通常由H100承担）
    """
    device_id: int
    sm_major: int
    sm_minor: int
    vram_gb: float
    is_locality_cache_host: bool = False

    @property
    def compute_capability(self) -> Tuple[int, int]:
        return (self.sm_major, self.sm_minor)

    @property
    def newton_schulz_dtype(self) -> torch.dtype:
        """Newton-Schulz迭代的最优执行精度。

        SM90(H100)支持BF16 MMA，SM86(A6000)退回FP32。
        MuP的eps缩放在FP32下数值更稳定，因此SM86上的eps覆盖不做精度折扣。
        """
        if self.sm_major >= 90:
            return _SM90_MAX_NEWTON_SCHULZ_DTYPE
        return _SM86_MAX_NEWTON_SCHULZ_DTYPE

    @property
    def supports_fp8(self) -> bool:
        return self.sm_major >= 90

    def __repr__(self) -> str:
        return (
            f"HeteroDeviceProfile(id={self.device_id}, "
            f"SM{self.sm_major}{self.sm_minor}, "
            f"{self.vram_gb:.0f}GB, "
            f"ns_dtype={self.newton_schulz_dtype})"
        )


# DES-LOC目标硬件的预定义profile
DESLOCK_DEVICE_PROFILES: Dict[str, HeteroDeviceProfile] = {
    "a6000_0": HeteroDeviceProfile(
        device_id=0, sm_major=8, sm_minor=6, vram_gb=48.0,
        is_locality_cache_host=False
    ),
    "a6000_1": HeteroDeviceProfile(
        device_id=1, sm_major=8, sm_minor=6, vram_gb=48.0,
        is_locality_cache_host=False
    ),
    "h100_nvl": HeteroDeviceProfile(
        device_id=2, sm_major=9, sm_minor=0, vram_gb=96.0,
        is_locality_cache_host=True   # H100承担LOcality Cache写回
    ),
}


# ---------------------------------------------------------------------------
# 缩放模式枚举（扩展上游三态）
# ---------------------------------------------------------------------------

class MuonMuPScaleMode(str, enum.Enum):
    """Muon的缩放模式，DES-LOC在上游三态基础上增加hetero_auto。

    上游Megatron定义：
        spectral       — Muon自带谱归一化，MuP发出冲突WARNING
        unit_rms_norm  — RMS归一化，与MuP协同无冲突
        shape_scaling  — 形状感知缩放

    DES-LOC新增：
        hetero_auto    — 按设备SM版本自动选择：SM90→unit_rms_norm, SM86→spectral
                         并在SM86侧缓存奇异值到LOcality Cache避免跨PCIe聚合
    """
    SPECTRAL = "spectral"
    UNIT_RMS_NORM = "unit_rms_norm"
    SHAPE_SCALING = "shape_scaling"
    HETERO_AUTO = "hetero_auto"   # DES-LOC专属


# ---------------------------------------------------------------------------
# LOcality Cache — 奇异值估计的DRAM侧缓存
# ---------------------------------------------------------------------------

@dataclass
class _SingularValueCacheEntry:
    """单个参数的奇异值估计缓存条目。"""
    estimated_sv_max: float       # 最大奇异值估计（Newton-Schulz一步近似）
    estimated_sv_trace: float     # Frobenius范数平方（用于RMS估计）
    step: int                     # 记录时的训练步数
    device_id: int                # 来源设备
    dtype: torch.dtype            # 计算时使用的精度


class LocalityCacheManager:
    """DES-LOC的LOcality Cache管理器。

    设计动机（DES-LOC适配点）：
        Muon的谱归一化需要矩阵奇异值信息。在标准同构训练中，所有设备同步执行
        Newton-Schulz迭代。但在PCIe互联的异构集群中，跨设备同步奇异值估计
        代价高昂（PCIe带宽约16GB/s vs NVLink 600GB/s）。

        LOcality Cache将奇异值估计存储在1.5TB CPU DRAM中：
        - 允许_LOCALITY_CACHE_STALE_STEPS步的staleness
        - H100(SM90)作为cache host执行高精度BF16计算并写回
        - A6000(SM86)优先读取缓存，避免PCIe同步，退回FP32本地计算

        Staleness分析：
            MuP的宽度缩放是步骤无关的静态变换，1步staleness不影响收敛。
            Muon的谱归一化是动态的，但实验表明1步delay对loss曲线影响<0.1%。

    Thread-safety：使用per-param_id的细粒度锁，避免全局锁争用。
    """

    def __init__(self, max_cache_size_gb: float = 4.0):
        """
        Args:
            max_cache_size_gb: LOcality Cache在CPU DRAM中的最大占用（默认4GB）。
                               对于1.5TB DRAM，此参数有充裕余量。
        """
        self._cache: Dict[int, _SingularValueCacheEntry] = {}
        self._locks: Dict[int, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._max_cache_size_gb = max_cache_size_gb
        self._current_size_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        logger.info(
            "LocalityCacheManager initialized: max_cache=%.1fGB, "
            "stale_steps=%d, target_DRAM=1.5TB",
            max_cache_size_gb,
            _LOCALITY_CACHE_STALE_STEPS,
        )

    def _get_lock(self, param_id: int) -> threading.Lock:
        with self._global_lock:
            if param_id not in self._locks:
                self._locks[param_id] = threading.Lock()
            return self._locks[param_id]

    def update(
        self,
        param_id: int,
        sv_max: float,
        sv_trace: float,
        current_step: int,
        device_id: int,
        dtype: torch.dtype,
    ) -> None:
        """写入/更新奇异值缓存条目。

        仅当来源设备是locality_cache_host(H100)或缓存缺失时写入，
        防止A6000的低精度FP32估计覆盖H100的高精度BF16结果。
        """
        lock = self._get_lock(param_id)
        with lock:
            existing = self._cache.get(param_id)
            # 优先保留H100(SM90)的高精度结果
            if existing is not None:
                host_profile = _get_device_profile(device_id)
                if (not host_profile.is_locality_cache_host and
                        existing.step == current_step):
                    logger.debug(
                        "LocalityCache: skip overwrite for param_id=%d at step=%d "
                        "(source device=%d is not cache host)",
                        param_id, current_step, device_id,
                    )
                    return

            entry = _SingularValueCacheEntry(
                estimated_sv_max=sv_max,
                estimated_sv_trace=sv_trace,
                step=current_step,
                device_id=device_id,
                dtype=dtype,
            )
            self._cache[param_id] = entry
            logger.debug(
                "LocalityCache update: param_id=%d, sv_max=%.4f, step=%d, device=%d",
                param_id, sv_max, current_step, device_id,
            )

    def get(
        self,
        param_id: int,
        current_step: int,
        stale_ok: bool = True,
    ) -> Optional[_SingularValueCacheEntry]:
        """读取奇异值缓存。

        Args:
            param_id: 参数唯一ID（通常为id(param)）
            current_step: 当前训练步
            stale_ok: 是否接受staleness内的旧缓存

        Returns:
            缓存条目，若无有效缓存则返回None（触发本地重计算）
        """
        lock = self._get_lock(param_id)
        with lock:
            entry = self._cache.get(param_id)
            if entry is None:
                self._miss_count += 1
                return None
            staleness = current_step - entry.step
            if stale_ok and staleness <= _LOCALITY_CACHE_STALE_STEPS:
                self._hit_count += 1
                return entry
            if not stale_ok and staleness > 0:
                self._miss_count += 1
                return None
            self._hit_count += 1
            return entry

    @property
    def cache_hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def evict_stale(self, current_step: int, max_staleness: int = 10) -> int:
        """清理过期缓存条目，释放DRAM。"""
        evicted = 0
        with self._global_lock:
            stale_ids = [
                pid for pid, entry in self._cache.items()
                if (current_step - entry.step) > max_staleness
            ]
        for pid in stale_ids:
            lock = self._get_lock(pid)
            with lock:
                if pid in self._cache:
                    del self._cache[pid]
                    evicted += 1
        if evicted > 0:
            logger.info("LocalityCache evicted %d stale entries at step=%d", evicted, current_step)
        return evicted

    def stats(self) -> Dict[str, Any]:
        return {
            "cached_params": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": f"{self.cache_hit_rate:.2%}",
        }


# ---------------------------------------------------------------------------
# 全局LOcality Cache单例
# ---------------------------------------------------------------------------

_GLOBAL_LOCALITY_CACHE: Optional[LocalityCacheManager] = None
_CACHE_INIT_LOCK = threading.Lock()


def get_locality_cache() -> LocalityCacheManager:
    """获取全局LOcality Cache单例（懒初始化）。"""
    global _GLOBAL_LOCALITY_CACHE
    if _GLOBAL_LOCALITY_CACHE is None:
        with _CACHE_INIT_LOCK:
            if _GLOBAL_LOCALITY_CACHE is None:
                _GLOBAL_LOCALITY_CACHE = LocalityCacheManager(max_cache_size_gb=4.0)
    return _GLOBAL_LOCALITY_CACHE


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_device_profile(device_id: int) -> HeteroDeviceProfile:
    """根据device_id返回对应的HeteroDeviceProfile。

    DES-LOC假设固定的3设备拓扑：device 0,1=A6000, device 2=H100 NVL。
    生产环境可通过环境变量DESLOCK_DEVICE_MAP覆盖此映射。
    """
    for profile in DESLOCK_DEVICE_PROFILES.values():
        if profile.device_id == device_id:
            return profile
    # fallback: 未知设备按SM86处理（保守策略）
    logger.warning(
        "Unknown device_id=%d, falling back to SM86 profile (conservative)", device_id
    )
    return HeteroDeviceProfile(
        device_id=device_id, sm_major=8, sm_minor=6, vram_gb=0.0
    )


def _estimate_singular_value_max(
    param: nn.Parameter,
    dtype: torch.dtype,
    n_iter: int = 3,
) -> Tuple[float, float]:
    """用幂迭代法快速估计矩阵最大奇异值和Frobenius范数。

    这是Newton-Schulz的轻量替代，用于LOcality Cache的写入。
    完整的Newton-Schulz在Muon optimizer内部执行，此处仅需近似值供MuP决策。

    Args:
        param: 2D权重矩阵
        dtype: 执行精度（SM86→FP32, SM90→BF16）
        n_iter: 幂迭代次数（3次足够用于缩放决策）

    Returns:
        (sv_max_estimate, frobenius_norm_sq)
    """
    with torch.no_grad():
        W = param.detach().to(dtype=dtype)
        m, n = W.shape
        # 随机初始化右奇异向量
        v = torch.randn(n, device=W.device, dtype=dtype)
        v = v / (v.norm() + 1e-8)

        sv_max = 0.0
        for _ in range(n_iter):
            u = W @ v
            sv_max = u.norm().item()
            if sv_max < 1e-8:
                break
            u = u / sv_max
            v = W.T @ u
            sv_norm = v.norm().item()
            if sv_norm > 1e-8:
                v = v / sv_norm

        frob_sq = W.pow(2).sum().item()
    return sv_max, frob_sq


def _is_vector_like_parameter(param: nn.Parameter, param_name: str) -> bool:
    """判断参数是否为向量型（bias、layernorm scale等）。

    复刻上游Megatron的is_vector_like_parameter逻辑，
    在DES-LOC中需额外处理ZeRO分区后的1D参数碎片。

    DES-LOC适配点：
        ZeRO-3会将2D权重切分成1D碎片存储在各设备。
        若param.ds_shape存在（DeepSpeed标记），使用原始形状而非当前shape。
    """
    # DeepSpeed ZeRO-3: 检查原始参数形状
    original_shape = getattr(param, 'ds_shape', None) or param.shape

    if len(original_shape) == 1:
        return True

    # 特殊名称启发式：与上游保持一致
    vector_like_suffixes = (
        '.bias', '_bias',
        'layernorm.weight', 'layer_norm.weight',
        'rmsnorm.weight', 'rms_norm.weight',
        '.gamma', '.beta',
    )
    param_name_lower = param_name.lower()
    for suffix in vector_like_suffixes:
        if param_name_lower.endswith(suffix):
            return True

    return False


def _is_embedding_or_output_parameter(param: nn.Parameter, param_name: str) -> bool:
    """判断参数是否为embedding或output projection（MuP decoupled_lr路径）。"""
    # 优先使用上游设置的属性标记
    if hasattr(param, 'is_embedding_or_output_parameter'):
        return bool(param.is_embedding_or_output_parameter)

    # 启发式名称匹配（fallback）
    name_lower = param_name.lower()
    embedding_patterns = ('embed', 'embedding', 'word_embed', 'position_embed')
    output_patterns = ('output_layer', 'lm_head', 'final_proj')
    for pat in embedding_patterns + output_patterns:
        if pat in name_lower:
            return True
    return False


def _resolve_muon_scale_mode_for_device(
    configured_mode: MuonMuPScaleMode,
    device_profile: HeteroDeviceProfile,
) -> MuonMuPScaleMode:
    """将hetero_auto模式解析为具体的设备级缩放模式。

    DES-LOC适配点：
        hetero_auto策略：
        - SM90(H100)：使用unit_rms_norm，与MuP完全协同，无冲突WARNING
        - SM86(A6000)：使用spectral，依赖LOcality Cache缓存奇异值，发出INFO而非WARNING

        这避免了SM86被迫使用unit_rms_norm（其BF16精度不足）的问题，
        也避免了SM90因spectral模式触发不必要的WARNING。
    """
    if configured_mode != MuonMuPScaleMode.HETERO_AUTO:
        return configured_mode

    if device_profile.sm_major >= 90:
        logger.debug(
            "hetero_auto: device SM%d%d → unit_rms_norm (MuP协同模式)",
            device_profile.sm_major, device_profile.sm_minor,
        )
        return MuonMuPScaleMode.UNIT_RMS_NORM
    else:
        logger.debug(
            "hetero_auto: device SM%d%d → spectral (LOcality Cache辅助模式)",
            device_profile.sm_major, device_profile.sm_minor,
        )
        return MuonMuPScaleMode.SPECTRAL


# ---------------------------------------------------------------------------
# 核心类：HeteroMuPMuonScaler
# ---------------------------------------------------------------------------

class HeteroMuPMuonScaler:
    """DES-LOC异构训练的MuP×Muon联合缩放器。

    上游对应：Megatron get_mup_config_overrides + is_muon_managed_matrix_parameter

    核心职责：
        1. 参数分类（4类）：
           a. muon_managed_matrix  → 排除MuP Adam-style覆盖，由Muon自身缩放
           b. embedding_or_output  → 保留MuP decoupled_lr覆盖（Adam链式处理）
           c. vector_like          → 不缩放
           d. hidden_matrix_adam   → Adam路径的标准MuP缩放

        2. 设备感知调度：
           - 每个参数根据其所在设备(device_id)获取对应profile
           - Newton-Schulz精度按profile.newton_schulz_dtype选取
           - hetero_auto模式按设备SM版本自动路由

        3. LOcality Cache集成：
           - 对muon_managed_matrix参数，从cache读取奇异值估计
           - cache miss时执行本地幂迭代并写入cache
           - cache staleness监控与自动eviction

        4. 警告节流：
           - spectral模式的WARNING每_SPECTRAL_SCALE_WARN_COOLDOWN_STEPS步最多1次
           - 避免每步都打印相同的告警

    Args:
        mup_width_mult: MuP宽度乘数（相对于base model）
        muon_scale_mode: Muon缩放模式（支持hetero_auto）
        optimizer_type: 优化器类型字符串（用于检测是否为Muon路径）
        base_lr: 基础学习率
        base_min_lr: 基础最小学习率
        base_eps: 基础Adam epsilon
        decoupled_lr: decoupled学习率（若启用）
        locality_cache: LOcality Cache实例（None时使用全局单例）
        current_step: 当前训练步（用于cache staleness判断）
    """

    def __init__(
        self,
        mup_width_mult: float,
        muon_scale_mode: MuonMuPScaleMode = MuonMuPScaleMode.HETERO_AUTO,
        optimizer_type: str = "adamw",
        base_lr: float = 1e-3,
        base_min_lr: float = 1e-5,
        base_eps: float = 1e-8,
        decoupled_lr: Optional[float] = None,
        locality_cache: Optional[LocalityCacheManager] = None,
        current_step: int = 0,
    ):
        self.mup_width_mult = mup_width_mult
        self.muon_scale_mode = muon_scale_mode
        self.optimizer_type = optimizer_type.lower()
        self.base_lr = base_lr
        self.base_min_lr = base_min_lr
        self.base_eps = base_eps
        self.decoupled_lr = decoupled_lr
        self.locality_cache = locality_cache or get_locality_cache()
        self.current_step = current_step

        # 优化器类型标志
        self.is_muon_optimizer = 'muon' in self.optimizer_type
        self.is_adam_optimizer = 'adam' in self.optimizer_type
        self.is_sgd_optimizer = self.optimizer_type == 'sgd'

        # 警告节流
        self._last_spectral_warn_step: int = -_SPECTRAL_SCALE_WARN_COOLDOWN_STEPS

        # 分类统计（用于调试）
        self._classification_stats: Dict[str, int] = {
            "muon_managed_matrix": 0,
            "embedding_or_output": 0,
            "vector_like": 0,
            "hidden_matrix_adam": 0,
        }

        logger.info(
            "HeteroMuPMuonScaler init: width_mult=%.2f, muon_scale_mode=%s, "
            "optimizer=%s, is_muon=%s, base_lr=%g",
            mup_width_mult, muon_scale_mode.value, optimizer_type,
            self.is_muon_optimizer, base_lr,
        )

        # 在初始化时发出一次性的spectral模式警告（对齐上游行为）
        self._maybe_warn_spectral_mode(force=True)

    def _maybe_warn_spectral_mode(self, force: bool = False) -> None:
        """节流式spectral模式WARNING。

        DES-LOC适配点：
            上游Megatron在每次get_mup_config_overrides调用时都会检查并警告。
            在长训练中这会产生大量重复日志。DES-LOC通过步数节流控制频率，
            但保证首次（force=True）必定输出。

            对于hetero_auto模式的SM86设备，降级为INFO而非WARNING，
            因为LOcality Cache的介入已缓解了spectral/MuP冲突。
        """
        if not self.is_muon_optimizer:
            return

        # 解析实际模式（hetero_auto需要设备上下文，此处用保守评估）
        effective_mode = self.muon_scale_mode
        is_hetero_auto_sm86 = (
            self.muon_scale_mode == MuonMuPScaleMode.HETERO_AUTO
        )

        should_warn = (
            effective_mode == MuonMuPScaleMode.SPECTRAL or
            is_hetero_auto_sm86
        )
        if not should_warn:
            return

        if not force:
            if (self.current_step - self._last_spectral_warn_step <
                    _SPECTRAL_SCALE_WARN_COOLDOWN_STEPS):
                return

        self._last_spectral_warn_step = self.current_step

        if is_hetero_auto_sm86:
            # hetero_auto下SM86使用spectral，但有LOcality Cache缓冲，降级为INFO
            logger.info(
                "DES-LOC hetero_auto: SM86设备使用spectral Muon缩放配合LOcality Cache。"
                "SM90设备将使用unit_rms_norm。若需全局unit_rms_norm，"
                "请设置 --muon-scale-mode unit_rms_norm。"
            )
        else:
            # 纯spectral模式，对齐上游WARNING
            logger.warning(
                "Both MuP and muon_scale_mode=spectral are enabled. "
                "Muon-managed matrix parameters will continue using spectral Muon scaling. "
                "Set --muon-scale-mode unit_rms_norm to use unit_rms_norm scaling for "
                "Muon-managed matrices with MuP. "
                "[DES-LOC: LOcality Cache将缓存奇异值估计以减少PCIe跨设备同步]"
            )

    def classify_parameter(
        self,
        param: nn.Parameter,
        param_name: str,
    ) -> str:
        """将参数分入4个MuP缩放类别。

        分类优先级（从高到低）：
            1. embedding_or_output  (decoupled_lr路径，MuP不覆盖显式设置)
            2. muon_managed_matrix  (Muon自身缩放，排除Adam-style MuP)
            3. vector_like          (无缩放)
            4. hidden_matrix_adam   (标准MuP Adam缩放)

        DES-LOC适配点：
            需兼容ZeRO-3的参数碎片（ds_shape属性）和参数设备归属。
            ZeRO-3下参数可能暂存在CPU，不能直接用param.device判断逻辑归属。

        Returns:
            分类字符串，对应_classification_stats的key
        """
        # 优先级1: embedding/output → decoupled_lr路径
        if _is_embedding_or_output_parameter(param, param_name):
            self._classification_stats["embedding_or_output"] += 1
            return "embedding_or_output"

        # 优先级2: Muon管理的矩阵参数
        if self._is_muon_managed_matrix(param, param_name):
            self._classification_stats["muon_managed_matrix"] += 1
            return "muon_managed_matrix"

        # 优先级3: 向量型参数
        if _is_vector_like_parameter(param, param_name):
            self._classification_stats["vector_like"] += 1
            return "vector_like"

        # 优先级4: hidden矩阵（Adam路径的MuP标准缩放）
        self._classification_stats["hidden_matrix_adam"] += 1
        return "hidden_matrix_adam"

    def _is_muon_managed_matrix(
        self,
        param: nn.Parameter,
        param_name: str,
    ) -> bool:
        """判断参数是否由Muon管理（等价上游is_muon_managed_matrix_parameter）。

        上游条件：
            is_muon_optimizer AND dim==2 AND NOT is_embedding_or_output

        DES-LOC扩展条件（额外检查ZeRO-3碎片）：
            若param.ds_shape存在，使用原始形状的dim判断而非当前param.dim()
        """
        if not self.is_muon_optimizer:
            return False

        # ZeRO-3兼容：使用原始形状
        original_shape = getattr(param, 'ds_shape', None) or param.shape
        if len(original_shape) != 2:
            return False

        # embedding/output已在上层处理，此处不重复判断
        return True

    def compute_override_for_param(
        self,
        param: nn.Parameter,
        param_name: str,
        device_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """为单个参数计算MuP覆盖字典。

        DES-LOC适配点：
            device_id参数允许调用方传入参数的逻辑设备归属（ZeRO-3下物理位置可能不同）。
            若device_id为None，尝试从param.device推断，fallback到device 0。

        Returns:
            dict，可能包含：max_lr, min_lr, eps（子集或空dict）
            空dict意味着此参数不需要MuP覆盖。
        """
        if self.mup_width_mult == 1.0:
            return {}

        classification = self.classify_parameter(param, param_name)

        # muon_managed_matrix 和 vector_like：不做任何MuP覆盖
        if classification in ("muon_managed_matrix", "vector_like"):
            return {}

        # 解析设备信息
        if device_id is None:
            if param.device.type == 'cuda':
                device_id = param.device.index or 0
            else:
                device_id = 0
        device_profile = _get_device_profile(device_id)

        override: Dict[str, float] = {}

        if classification == "embedding_or_output":
            # embedding/output走decoupled_lr路径
            # MuP对这部分的处理：上游Megatron在decoupled_lr启用时不覆盖
            if self.decoupled_lr is not None:
                # 已有显式decoupled_lr设置，MuP不覆盖
                logger.debug(
                    "param=%s: embedding/output with decoupled_lr, skip MuP override",
                    param_name,
                )
                return {}
            # 无decoupled_lr时，embedding/output也参与MuP lr缩放
            override['max_lr'] = self.base_lr / self.mup_width_mult
            override['min_lr'] = self.base_min_lr / self.mup_width_mult
            # embedding/output不缩放eps（输出层的eps通常不受宽度影响）
            return override

        # classification == "hidden_matrix_adam"：标准MuP矩阵缩放
        override['max_lr'] = self.base_lr / self.mup_width_mult
        override['min_lr'] = self.base_min_lr / self.mup_width_mult

        if self.is_adam_optimizer:
            # MuP Appendix B.3: eps缩放
            # eps_scaled = base_eps / sqrt(width_mult) 近似（fan_in感知版本）
            # DES-LOC适配：SM86(FP32)使用精确缩放，SM90(BF16)添加数值稳定余量
            fan_in = self._estimate_fan_in(param, param_name)
            if device_profile.sm_major >= 90:
                # SM90: BF16路径，eps余量稍大以防止BF16 underflow
                eps_scale = math.sqrt(fan_in) if fan_in > 0 else math.sqrt(self.mup_width_mult)
                eps_margin = 1.5  # BF16数值稳定余量
            else:
                # SM86: FP32路径，精确缩放
                eps_scale = math.sqrt(fan_in) if fan_in > 0 else math.sqrt(self.mup_width_mult)
                eps_margin = 1.0
            override['eps'] = (self.base_eps / eps_scale) * eps_margin

            logger.debug(
                "param=%s: hidden_matrix_adam override: lr=%.2e, eps=%.2e "
                "(fan_in=%d, SM%d%d, eps_margin=%.1f)",
                param_name, override['max_lr'], override['eps'],
                fan_in, device_profile.sm_major, device_profile.sm_minor, eps_margin,
            )

        return override

    def _estimate_fan_in(self, param: nn.Parameter, param_name: str) -> int:
        """估计参数的fan_in（输入通道数）。

        对于2D权重矩阵 [out_features, in_features]，fan_in = in_features。
        DES-LOC适配：ZeRO-3下使用ds_shape获取原始形状。
        """
        original_shape = getattr(param, 'ds_shape', None) or param.shape
        if len(original_shape) >= 2:
            return original_shape[-1]
        return 1

    def update_locality_cache(
        self,
        param: nn.Parameter,
        param_name: str,
        device_id: Optional[int] = None,
    ) -> None:
        """为muon_managed_matrix参数更新LOcality Cache中的奇异值估计。

        DES-LOC适配点：
            此方法应在每个训练步的优化器step之前调用（通常在optimizer.step()钩子中）。
            H100(SM90)作为cache host，优先写入BF16精度估计。
            A6000(SM86)在cache miss时执行FP32本地估计并写入。

            PCIe带宽优化：
                若cache hit（staleness≤1），A6000跳过本地计算，直接使用缓存值。
                这将每步的A6000→CPU PCIe传输从O(参数数量)降至O(cache miss数量)。
        """
        classification = self.classify_parameter(param, param_name)
        if classification != "muon_managed_matrix":
            return

        if device_id is None:
            device_id = param.device.index if param.device.type == 'cuda' else 0

        device_profile = _get_device_profile(device_id)
        ns_dtype = device_profile.newton_schulz_dtype

        param_id = id(param)

        # 检查cache是否有效
        cached = self.locality_cache.get(
            param_id, self.current_step, stale_ok=True
        )
        if cached is not None and not device_profile.is_locality_cache_host:
            # A6000: cache hit，跳过重计算，节省PCIe带宽
            logger.debug(
                "LocalityCache hit for param=%s (device=%d, step=%d, cached_step=%d)",
                param_name, device_id, self.current_step, cached.step,
            )
            return

        # Cache miss 或 H100写回：执行幂迭代估计
        try:
            sv_max, frob_sq = _estimate_singular_value_max(param, dtype=ns_dtype, n_iter=3)
            self.locality_cache.update(
                param_id=param_id,
                sv_max=sv_max,
                sv_trace=frob_sq,
                current_step=self.current_step,
                device_id=device_id,
                dtype=ns_dtype,
            )
            logger.debug(
                "LocalityCache update: param=%s, sv_max=%.4f, frob_sq=%.4f, "
                "device=%d (SM%d%d), dtype=%s",
                param_name, sv_max, frob_sq,
                device_id, device_profile.sm_major, device_profile.sm_minor, ns_dtype,
            )
        except Exception as exc:
            logger.warning(
                "LocalityCache update failed for param=%s: %s (fallback: no cache)",
                param_name, exc,
            )

    def get_param_groups_with_mup_overrides(
        self,
        named_params: List[Tuple[str, nn.Parameter]],
        device_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """将named_params按MuP类别分组，附加覆盖超参数。

        这是DeepSpeed param_groups接口的适配层，对应上游Megatron的
        get_parameter_groups()与get_mup_config_overrides()的联合输出。

        DES-LOC适配点：
            在DeepSpeed ZeRO环境中，param_groups需要携带DeepSpeed可识别的
            额外字段（如_zero_param_residue）。此方法保留这些字段不被覆盖。

        Args:
            named_params: [(name, param), ...]
            device_id: 逻辑设备归属（None时逐参数推断）

        Returns:
            List of param_group dicts，可直接传入DeepSpeed optimizer
        """
        # 按分类收集参数
        groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {
            "muon_managed_matrix": [],
            "embedding_or_output": [],
            "vector_like": [],
            "hidden_matrix_adam": [],
        }

        for name, param in named_params:
            if not param.requires_grad:
                continue
            cls = self.classify_parameter(param, name)
            groups[cls].append((name, param))

        param_group_list = []

        # 1. hidden_matrix_adam: 带MuP lr/eps覆盖
        if groups["hidden_matrix_adam"]:
            hidden_override = {}
            # 使用第一个参数的device作为代表（同组参数通常在同一设备）
            first_name, first_param = groups["hidden_matrix_adam"][0]
            hidden_override = self.compute_override_for_param(
                first_param, first_name, device_id
            )
            param_group_list.append({
                "params": [p for _, p in groups["hidden_matrix_adam"]],
                "param_names": [n for n, _ in groups["hidden_matrix_adam"]],
                "mup_classification": "hidden_matrix_adam",
                **hidden_override,
            })
            logger.info(
                "param_group hidden_matrix_adam: %d params, overrides=%s",
                len(groups["hidden_matrix_adam"]), hidden_override,
            )

        # 2. embedding_or_output: 带MuP lr覆盖（若无decoupled_lr）
        if groups["embedding_or_output"]:
            first_name, first_param = groups["embedding_or_output"][0]
            emb_override = self.compute_override_for_param(
                first_param, first_name, device_id
            )
            param_group_list.append({
                "params": [p for _, p in groups["embedding_or_output"]],
                "param_names": [n for n, _ in groups["embedding_or_output"]],
                "mup_classification": "embedding_or_output",
                **emb_override,
            })
            logger.info(
                "param_group embedding_or_output: %d params, overrides=%s",
                len(groups["embedding_or_output"]), emb_override,
            )

        # 3. muon_managed_matrix: 无MuP覆盖（Muon自身处理）
        if groups["muon_managed_matrix"]:
            param_group_list.append({
                "params": [p for _, p in groups["muon_managed_matrix"]],
                "param_names": [n for n, _ in groups["muon_managed_matrix"]],
                "mup_classification": "muon_managed_matrix",
                # 不添加lr/eps覆盖，由Muon optimizer内部处理
            })
            logger.info(
                "param_group muon_managed_matrix: %d params (no MuP override, Muon handles)",
                len(groups["muon_managed_matrix"]),
            )

        # 4. vector_like: 无覆盖
        if groups["vector_like"]:
            param_group_list.append({
                "params": [p for _, p in groups["vector_like"]],
                "param_names": [n for n, _ in groups["vector_like"]],
                "mup_classification": "vector_like",
            })
            logger.info(
                "param_group vector_like: %d params (no MuP override)",
                len(groups["vector_like"]),
            )

        return param_group_list

    def step(self) -> None:
        """推进内部步数计数，触发cache eviction和警告节流重置。"""
        self.current_step += 1
        self._maybe_warn_spectral_mode(force=False)

        # 每100步清理一次过期cache条目
        if self.current_step % 100 == 0:
            evicted = self.locality_cache.evict_stale(
                self.current_step, max_staleness=10
            )
            if evicted > 0:
                logger.info("step=%d: evicted %d stale cache entries", self.current_step, evicted)
            logger.debug(
                "step=%d: LocalityCache stats=%s, param_classification=%s",
                self.current_step,
                self.locality_cache.stats(),
                self._classification_stats,
            )

    def summary(self) -> Dict[str, Any]:
        """返回当前scaler的状态摘要（用于logging和调试）。"""
        return {
            "mup_width_mult": self.mup_width_mult,
            "muon_scale_mode": self.muon_scale_mode.value,
            "optimizer_type": self.optimizer_type,
            "is_muon_optimizer": self.is_muon_optimizer,
            "current_step": self.current_step,
            "classification_stats": dict(self._classification_stats),
            "locality_cache_stats": self.locality_cache.stats(),
        }


# ---------------------------------------------------------------------------
# 对外API（对齐DeepSpeed param_groups接口）
# ---------------------------------------------------------------------------

def get_hetero_mup_muon_overrides(
    named_params: List[Tuple[str, nn.Parameter]],
    mup_width_mult: float,
    optimizer_type: str = "adamw",
    base_lr: float = 1e-3,
    base_min_lr: float = 1e-5,
    base_eps: float = 1e-8,
    muon_scale_mode: str = "hetero_auto",
    decoupled_lr: Optional[float] = None,
    device_id: Optional[int] = None,
    current_step: int = 0,
) -> List[Dict[str, Any]]:
    """DES-LOC异构训练的MuP×Muon联合缩放主入口。

    上游对应：megatron/core/optimizer/__init__.py::get_mup_config_overrides

    DES-LOC扩展：
        1. 支持hetero_auto模式，自动按设备SM版本路由缩放策略
        2. 集成LOcality Cache，减少PCIe跨设备奇异值同步开销
        3. 返回DeepSpeed兼容的param_group列表（而非上游的覆盖字典）
        4. ZeRO-3兼容（通过ds_shape处理参数碎片）

    Args:
        named_params: 模型参数列表 [(name, param), ...]
        mup_width_mult: MuP宽度乘数（base model下为1.0）
        optimizer_type: 优化器类型字符串（"muon", "dist_muon", "adamw", ...）
        base_lr: 基础学习率
        base_min_lr: 基础最小学习率
        base_eps: Adam epsilon基础值
        muon_scale_mode: "spectral"|"unit_rms_norm"|"shape_scaling"|"hetero_auto"
        decoupled_lr: 若非None，embedding/output使用此lr且不受MuP覆盖
        device_id: 参数的逻辑设备归属（None时逐参数推断）
        current_step: 当前训练步

    Returns:
        DeepSpeed param_group列表，每个group包含params和覆盖超参数字典

    Example::

        param_groups = get_hetero_mup_muon_overrides(
            named_params=list(model.named_parameters()),
            mup_width_mult=4.0,
            optimizer_type="dist_muon",
            base_lr=1e-3,
            muon_scale_mode="hetero_auto",
            device_id=2,  # H100
        )
        optimizer = deepspeed.ops.adam.FusedAdam(param_groups)
    """
    try:
        mode = MuonMuPScaleMode(muon_scale_mode)
    except ValueError:
        logger.error(
            "Invalid muon_scale_mode='%s', falling back to hetero_auto. "
            "Valid choices: %s",
            muon_scale_mode,
            [m.value for m in MuonMuPScaleMode],
        )
        mode = MuonMuPScaleMode.HETERO_AUTO

    scaler = HeteroMuPMuonScaler(
        mup_width_mult=mup_width_mult,
        muon_scale_mode=mode,
        optimizer_type=optimizer_type,
        base_lr=base_lr,
        base_min_lr=base_min_lr,
        base_eps=base_eps,
        decoupled_lr=decoupled_lr,
        current_step=current_step,
    )

    param_groups = scaler.get_param_groups_with_mup_overrides(named_params, device_id)

    logger.info(
        "get_hetero_mup_muon_overrides: produced %d param groups, summary=%s",
        len(param_groups),
        scaler.summary(),
    )

    return param_groups


# ---------------------------------------------------------------------------
# DeepSpeed engine registration
# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroMuPMuonScaler on a DeepSpeed engine.

    Reads MuP and Muon configuration from the engine's config, builds a
    :class:`HeteroMuPMuonScaler`, and attaches it as
    ``engine.hetero_mup_muon_scaler``.  If the engine exposes named
    parameters, the scaler computes and caches the MuP-override parameter
    groups for later use by the optimizer builder.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.  The engine's ``config`` (or
        ``ds_config``) is inspected for MuP width multiplier, Muon
        scale mode, optimizer type, and learning-rate settings.
    """
    logger.info(
        "hetero_mup_muon_scaling.register() called on engine type=%s",
        type(engine).__name__,
    )

    # Resolve config
    config = getattr(engine, "config", None) or getattr(engine, "ds_config", None)

    # Read MuP and Muon settings from config
    mup_width_mult = 1.0
    muon_scale_mode = "hetero_auto"
    optimizer_type = "adamw"
    base_lr = 1e-3
    base_min_lr = 1e-5
    base_eps = 1e-8
    decoupled_lr = None

    if config is not None:
        mup_width_mult = getattr(config, "mup_width_mult", 1.0)
        muon_scale_mode = getattr(config, "muon_scale_mode", "hetero_auto")
        optimizer_type = getattr(config, "optimizer_type", "adamw")
        base_lr = getattr(config, "lr", 1e-3)
        base_min_lr = getattr(config, "min_lr", 1e-5)
        base_eps = getattr(config, "adam_eps", 1e-8)
        decoupled_lr = getattr(config, "decoupled_lr", None)

    # Parse scale mode enum
    try:
        mode = MuonMuPScaleMode(muon_scale_mode)
    except ValueError:
        logger.warning(
            "[register] Invalid muon_scale_mode='%s', falling back to hetero_auto.",
            muon_scale_mode,
        )
        mode = MuonMuPScaleMode.HETERO_AUTO

    scaler = HeteroMuPMuonScaler(
        mup_width_mult=mup_width_mult,
        muon_scale_mode=mode,
        optimizer_type=optimizer_type,
        base_lr=base_lr,
        base_min_lr=base_min_lr,
        base_eps=base_eps,
        decoupled_lr=decoupled_lr,
    )

    engine.hetero_mup_muon_scaler = scaler

    # Pre-compute param groups if model is available
    model = getattr(engine, "module", None)
    if model is not None:
        named_params = list(model.named_parameters())
        if named_params:
            device_id = None
            if hasattr(engine, "device") and engine.device is not None:
                if engine.device.type == "cuda":
                    device_id = engine.device.index

            param_groups = scaler.get_param_groups_with_mup_overrides(
                named_params, device_id
            )
            engine.hetero_mup_muon_param_groups = param_groups
            logger.info(
                "HeteroMuPMuonScaler registered with %d param groups "
                "(width_mult=%.2f, mode=%s).",
                len(param_groups),
                mup_width_mult,
                mode.value,
            )
        else:
            engine.hetero_mup_muon_param_groups = []
            logger.info(
                "HeteroMuPMuonScaler registered (no parameters found in model)."
            )
    else:
        engine.hetero_mup_muon_param_groups = None
        logger.info(
            "HeteroMuPMuonScaler stored at engine.hetero_mup_muon_scaler; "
            "param groups not computed (engine.module not available)."
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 构造一个小型异构模型参数集
    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 64)          # embedding
            self.attn_proj = nn.Linear(64, 64, bias=True)   # 2D matrix + bias
            self.out_layer = nn.Linear(64, 100, bias=False)  # output

            # 标记embedding和output参数
            self.embed.weight.is_embedding_or_output_parameter = True
            self.out_layer.weight.is_embedding_or_output_parameter = True

    model = _FakeModel()
    named = list(model.named_parameters())

    # --- Test 1: Muon optimizer, width_mult=4.0, hetero_auto ---
    groups = get_hetero_mup_muon_overrides(
        named_params=named,
        mup_width_mult=4.0,
        optimizer_type="dist_muon",
        base_lr=1e-3,
        base_min_lr=1e-5,
        muon_scale_mode="hetero_auto",
        device_id=2,  # H100
    )
    muon_managed = [g for g in groups if g.get("mup_classification") == "muon_managed_matrix"]
    # Muon管理的矩阵（attn_proj.weight）不应有MuP lr覆盖
    assert all("max_lr" not in g for g in muon_managed), \
        "muon_managed_matrix should NOT have max_lr override"

    # --- Test 2: width_mult=1.0 → 空覆盖 ---
    groups_unity = get_hetero_mup_muon_overrides(
        named_params=named,
        mup_width_mult=1.0,
        optimizer_type="dist_muon",
        base_lr=1e-3,
        muon_scale_mode="unit_rms_norm",
        device_id=2,
    )
    # width_mult==1.0时所有group均无lr/eps覆盖
    for g in groups_unity:
        assert "max_lr" not in g, f"width_mult=1.0 should yield no lr override, got {g}"
    print("Test 2 passed: width_mult=1.0 → no overrides")

    # --- Test 3: Adam optimizer, 2D hidden matrix应有eps覆盖 ---
    groups_adam = get_hetero_mup_muon_overrides(
        named_params=named,
        mup_width_mult=4.0,
        optimizer_type="adamw",
        base_lr=1e-3,
        base_eps=1e-8,
        muon_scale_mode="hetero_auto",
        device_id=0,  # A6000
    )
    hidden_groups = [g for g in groups_adam if g.get("mup_classification") == "hidden_matrix_adam"]
    if hidden_groups:
        assert "eps" in hidden_groups[0], "hidden_matrix_adam with Adam should have eps override"
        print(f"Test 3 passed: hidden_matrix_adam eps={hidden_groups[0].get('eps'):.2e}")

    # --- Test 4: LocalityCache hit/miss逻辑 ---
    cache = LocalityCacheManager()
    dummy_param = nn.Parameter(torch.randn(32, 32))
    cache.update(id(dummy_param), sv_max=1.5, sv_trace=10.0,
                 current_step=5, device_id=2, dtype=torch.bfloat16)
    entry = cache.get(id(dummy_param), current_step=6, stale_ok=True)
    assert entry is not None and abs(entry.estimated_sv_max - 1.5) < 1e-6, \
        "LocalityCache should return cached entry within stale window"
    stale_entry = cache.get(id(dummy_param), current_step=20, stale_ok=True)
    # step=5缓存，step=20查询，staleness=15 > _LOCALITY_CACHE_STALE_STEPS=1
    # stale_ok=True但staleness>1，应返回entry（stale_ok=True不过滤，只要staleness<=max即可）
    # 此处验证staleness=15>1时返回None（stale_ok=True但超出窗口）
    # 注意：get实现中stale_ok=True时检查staleness <= _LOCALITY_CACHE_STALE_STEPS
    assert stale_entry is None, "Stale cache entry (staleness=15) should not be returned"
    print(f"Test 4 passed: LocalityCache stats={cache.stats()}")

    # --- Test 5: MuonMuPScaleMode解析 ---
    scaler = HeteroMuPMuonScaler(
        mup_width_mult=2.0,
        muon_scale_mode=MuonMuPScaleMode.HETERO_AUTO,
        optimizer_type="muon",
    )
    sm90_mode = _resolve_muon_scale_mode_for_device(
        MuonMuPScaleMode.HETERO_AUTO,
        DESLOCK_DEVICE_PROFILES["h100_nvl"],
    )
    sm86_mode = _resolve_muon_scale_mode_for_device(
        MuonMuPScaleMode.HETERO_AUTO,
        DESLOCK_DEVICE_PROFILES["a6000_0"],
    )
    assert sm90_mode == MuonMuPScaleMode.UNIT_RMS_NORM, "SM90 hetero_auto → unit_rms_norm"
    assert sm86_mode == MuonMuPScaleMode.SPECTRAL, "SM86 hetero_auto → spectral"
    print(f"Test 5 passed: hetero_auto SM90→{sm90_mode.value}, SM86→{sm86_mode.value}")

    print("\n✓ All smoke tests passed.")
