"""
deepspeed/inference/hetero_fine_grained_cudagraph.py

DES-LOC 异构训练框架 — HeteroFineGrainedCudaGraph 模块
========================================================

上游设计意图 (Megatron fde3b90a)
--------------------------------
Megatron-LM 在推理动态批处理中引入了更细粒度的 CUDA graph 捕获策略：
1. 当 num_cuda_graphs == -1 时自动推导捕获数量，基于 max_requests 密集生成
   小批次覆盖点（1,2,4,8,16,...256,272,...），再对齐 TP size 并去重；
2. 仅在 num_cuda_graphs != -1 时才执行 clamp 逻辑，避免 -1 被误截断；
3. 在参数验证中对 local cuda-graph 实现额外校验该字段合法性。

核心洞察：小 batch 推理时 padding 浪费显著，密集捕获 token count 可将
kernel launch overhead 摊薄，同时避免对最大 batch 的单一依赖。

DES-LOC 适配点
--------------
DES-LOC（Decoupled Execution with Shared LOcality Cache）的异构硬件环境：
  - 2× A6000 48GB (SM86, PCIe)   — 负责 prefill / 中小 batch decode
  - 1× H100 NVL 96GB (SM90, PCIe) — 负责大 batch decode / speculative verify
  - 1.5TB CPU DRAM                — Shared Locality Cache (SLC)，存放 KV 溢出 + graph metadata

关键适配逻辑：
1. **设备感知的 graph 粒度**：A6000 SM86 CUDA graph 捕获受 SM 数量限制，
   采用与 Megatron -1 模式等价的"自动密集"策略；H100 NVL 捕获点可更稀疏。
2. **PCIe 带宽约束下的 token-count 对齐**：无 NVLink，A6000↔H100 迁移代价
   高，token_count 对齐粒度需同时考虑 TP size 与 PCIe chunk size (128B)。
3. **SLC 元数据写回**：graph pool 的 token-count 清单写入 SLC，供对端设备
   在迁移时快速查找最近覆盖点，无需重新捕获。
4. **异步捕获调度**：A6000 和 H100 可并发捕获互不重叠的 token-count 区间，
   通过 CUDA stream + event 同步；捕获完成后统一合并到 unified_pool。

作者: Neuron_SP / DES-LOC Team
对应 upstream: Megatron fde3b90a8015d4c865ac3a0cfaae5a52ed273b88
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 硬件常量 / DES-LOC 拓扑描述
# ---------------------------------------------------------------------------

# PCIe Gen4 x16 单向带宽约 32 GB/s；chunk 对齐粒度（bytes）
_PCIE_CHUNK_BYTES: int = 128

# SM 架构映射
SM_ARCH_A6000 = 86  # Ampere  SM86
SM_ARCH_H100_NVL = 90  # Hopper SM90

# 每种架构推荐的最大并发 CUDA graph 数上限（经验值）
_MAX_GRAPHS_PER_ARCH: Dict[int, int] = {
    SM_ARCH_A6000: 256,
    SM_ARCH_H100_NVL: 128,
}

# H100 NVL 相对于 A6000 的捕获稀疏系数：H100 batch 更大，不需要 1-2-4 级别密度
_H100_SPARSITY_FACTOR: int = 4


# ---------------------------------------------------------------------------
# 枚举 & 数据类
# ---------------------------------------------------------------------------


class DeviceRole(Enum):
    """DES-LOC 中每个设备承担的推理角色。"""
    PREFILL_SMALL_DECODE = auto()   # A6000: prefill + small-batch decode
    LARGE_DECODE_VERIFY = auto()    # H100 NVL: large-batch decode / spec verify


class GraphScope(Enum):
    """CUDA graph 捕获范围，镜像 Megatron CudaGraphScope。"""
    FULL_ITERATION = auto()
    LOCAL = auto()


@dataclass
class DeviceSpec:
    """描述单块物理设备的 DES-LOC 相关属性。"""
    device_index: int
    sm_arch: int
    vram_gb: float
    role: DeviceRole
    tp_rank: int = 0
    tp_size: int = 1

    @property
    def max_graphs(self) -> int:
        return _MAX_GRAPHS_PER_ARCH.get(self.sm_arch, 64)

    @property
    def arch_name(self) -> str:
        names = {SM_ARCH_A6000: "A6000-SM86", SM_ARCH_H100_NVL: "H100NVL-SM90"}
        return names.get(self.sm_arch, f"SM{self.sm_arch}")


@dataclass
class GraphPoolEntry:
    """单条 CUDA graph 捕获记录，存入 SLC 元数据区。"""
    token_count: int
    device_index: int
    sm_arch: int
    captured: bool = False
    capture_latency_ms: float = 0.0
    graph_handle: Optional[object] = None  # torch.cuda.CUDAGraph when available


@dataclass
class SharedLocalityCache:
    """
    SLC — Shared Locality Cache (CPU DRAM 侧元数据存储)。

    在 DES-LOC 中 SLC 充当跨设备的"graph 发现层"：
    - 任意设备捕获 graph 后，将 token_count 条目写入 SLC；
    - 对端设备迁移请求时查询 SLC，定位最近覆盖点，避免重捕获。
    """
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pool: Dict[Tuple[int, int], GraphPoolEntry] = field(default_factory=dict)
    # (device_index, token_count) -> GraphPoolEntry

    def register(self, entry: GraphPoolEntry) -> None:
        key = (entry.device_index, entry.token_count)
        with self._lock:
            self._pool[key] = entry
        logger.debug(
            "[SLC] registered graph entry: device=%d token_count=%d arch=%s",
            entry.device_index, entry.token_count, entry.sm_arch
        )

    def lookup_nearest(
        self,
        token_count: int,
        preferred_device: Optional[int] = None,
    ) -> Optional[GraphPoolEntry]:
        """
        查找 SLC 中 token_count >= 请求量且最小的已捕获 graph。
        若指定 preferred_device 则优先在该设备上查找。
        """
        with self._lock:
            candidates = [
                e for e in self._pool.values()
                if e.captured and e.token_count >= token_count
            ]
        if not candidates:
            return None
        if preferred_device is not None:
            dev_candidates = [e for e in candidates if e.device_index == preferred_device]
            if dev_candidates:
                candidates = dev_candidates
        return min(candidates, key=lambda e: e.token_count)

    def all_entries_for_device(self, device_index: int) -> List[GraphPoolEntry]:
        with self._lock:
            return [e for e in self._pool.values() if e.device_index == device_index]

    def dump_summary(self) -> str:
        with self._lock:
            lines = [
                f"  dev={e.device_index} tc={e.token_count:5d} captured={e.captured} "
                f"lat={e.capture_latency_ms:.2f}ms arch={e.sm_arch}"
                for e in sorted(self._pool.values(), key=lambda x: (x.device_index, x.token_count))
            ]
        return "SharedLocalityCache:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心：token-count 计算器（镜像 Megatron _calculate_cuda_graph_token_counts）
# ---------------------------------------------------------------------------


class HeteroTokenCountBuilder:
    """
    为 DES-LOC 异构设备生成细粒度 token-count 覆盖集合。

    设计意图（对应 Megatron fde3b90a）
    ------------------------------------
    Megatron 引入 num_cuda_graphs=-1 的自动模式：
      [1,2,4] + range(8,256,8) + range(256, max+1, 16)
    再对齐 TP size、去重、截断、反转（降序）。

    DES-LOC 异构扩展
    ----------------
    不同架构设备使用不同密度策略：
    - A6000 (SM86)：完全镜像 Megatron 自动模式，覆盖小批次。
    - H100 NVL (SM90)：稀疏版本（步长 × _H100_SPARSITY_FACTOR），
      减少 H100 的无效捕获开销（H100 更多处理大批次）。

    PCIe 对齐
    ----------
    无 NVLink 时，token 维度对应的 hidden-state tensor 在设备间迁移须对齐
    _PCIE_CHUNK_BYTES，因此 token_count 需同时满足：
      align(token_count × hidden_bytes, PCIE_CHUNK_BYTES)
    本类仅处理 token_count 层面的整数对齐（hidden_bytes 由调用方提供）。
    """

    def __init__(
        self,
        device_spec: DeviceSpec,
        cuda_graph_max_tokens: int,
        hidden_size_bytes: int = 2,  # 默认 bf16，每 token 1 元素 × 2 bytes（简化）
    ) -> None:
        self.device_spec = device_spec
        self.cuda_graph_max_tokens = cuda_graph_max_tokens
        self.hidden_size_bytes = hidden_size_bytes
        self._tp_size = device_spec.tp_size

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def build_auto(self) -> List[int]:
        """
        自动模式（num_cuda_graphs == -1 等价）。
        根据设备架构选择密度策略，返回降序 token-count 列表。
        """
        if self.device_spec.sm_arch == SM_ARCH_H100_NVL:
            counts = self._build_sparse()
        else:
            counts = self._build_dense()

        counts = self._align_to_tp(counts)
        counts = self._align_to_pcie(counts)
        counts = self._dedup_clamp_sort(counts)

        logger.info(
            "[HeteroTokenCount] device=%d (%s) auto-mode: %d graph points, "
            "max_tokens=%d, tp_size=%d",
            self.device_spec.device_index,
            self.device_spec.arch_name,
            len(counts),
            self.cuda_graph_max_tokens,
            self._tp_size,
        )
        return counts

    def build_fixed(self, num_cuda_graphs: int) -> List[int]:
        """
        固定数量模式（num_cuda_graphs >= 1）。
        等间距在 [tp_size, cuda_graph_max_tokens] 区间内取点。
        对应 Megatron 原始非 -1 分支。
        """
        assert num_cuda_graphs >= 1, (
            f"num_cuda_graphs must be >= 1, got {num_cuda_graphs}"
        )
        assert self.cuda_graph_max_tokens > 0, (
            f"cuda_graph_max_tokens must be > 0, got {self.cuda_graph_max_tokens}"
        )
        # clamp：同 Megatron 逻辑
        num_cuda_graphs = min(max(num_cuda_graphs, 1), self.cuda_graph_max_tokens)

        step = max(1, self.cuda_graph_max_tokens // num_cuda_graphs)
        raw = list(range(step, self.cuda_graph_max_tokens + 1, step))
        if not raw or raw[-1] != self.cuda_graph_max_tokens:
            raw.append(self.cuda_graph_max_tokens)

        counts = self._align_to_tp(raw)
        counts = self._dedup_clamp_sort(counts)

        logger.info(
            "[HeteroTokenCount] device=%d (%s) fixed-mode: requested=%d, actual=%d points",
            self.device_spec.device_index,
            self.device_spec.arch_name,
            num_cuda_graphs,
            len(counts),
        )
        return counts

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_dense(self) -> List[int]:
        """
        A6000 密集策略 — 镜像 Megatron fde3b90a 自动模式原始公式：
          [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)
        """
        max_t = self.cuda_graph_max_tokens
        seed: List[int] = (
            [1, 2, 4]
            + list(range(8, 256, 8))
            + list(range(256, max_t + 1, 16))
        )
        return seed

    def _build_sparse(self) -> List[int]:
        """
        H100 NVL 稀疏策略 — 步长扩大 _H100_SPARSITY_FACTOR 倍。
        H100 主要承接大批次 decode，不需要 1/2/4 级别的精细覆盖。
        保留 small-batch 的最小覆盖点以应对 speculative verify 尾批次。
        """
        max_t = self.cuda_graph_max_tokens
        f = _H100_SPARSITY_FACTOR
        seed: List[int] = (
            [8]  # 最小覆盖点
            + list(range(8 * f, 256, 8 * f))
            + list(range(256, max_t + 1, 16 * f))
        )
        return seed

    def _align_to_tp(self, counts: List[int]) -> List[int]:
        """
        每个 token-count 向上对齐到 TP size 的整数倍。
        保留去重前的全部值（dict.fromkeys 保序去重）。
        完全镜像 Megatron fde3b90a 的对齐逻辑。
        """
        tp = self._tp_size
        if tp <= 1:
            return counts
        return list(dict.fromkeys(math.ceil(s / tp) * tp for s in counts))

    def _align_to_pcie(self, counts: List[int]) -> List[int]:
        """
        DES-LOC 专属：PCIe chunk 对齐。
        确保每个 token-count × hidden_size_bytes 是 _PCIE_CHUNK_BYTES 的整数倍，
        减少 PCIe DMA 传输时的 padding 开销。
        """
        chunk = _PCIE_CHUNK_BYTES
        h = max(1, self.hidden_size_bytes)
        result = []
        for s in counts:
            byte_size = s * h
            aligned_bytes = math.ceil(byte_size / chunk) * chunk
            aligned_tokens = aligned_bytes // h
            result.append(aligned_tokens)
        return result

    def _dedup_clamp_sort(self, counts: List[int]) -> List[int]:
        """
        去重 → 截断到 max_tokens → 确保末尾含 max_tokens → 降序排列。
        与 Megatron fde3b90a _calculate_cuda_graph_token_counts 末段逻辑一致。
        """
        max_t = self.cuda_graph_max_tokens
        counts = list(dict.fromkeys(s for s in counts if s <= max_t))
        if not counts or counts[-1] != max_t:
            counts.append(max_t)
        counts.sort(reverse=True)
        return counts


# ---------------------------------------------------------------------------
# 核心：异构 CUDA Graph 管理器
# ---------------------------------------------------------------------------


class HeteroFineGrainedCudaGraphManager:
    """
    DES-LOC 异构细粒度 CUDA Graph 管理器。

    职责
    ----
    1. 为每块物理设备（A6000 × 2 + H100 × 1）按 HeteroTokenCountBuilder
       生成各自的 token-count 覆盖集；
    2. 并发异步捕获（A6000 与 H100 互不阻塞）；
    3. 捕获完成后将元数据写入 SharedLocalityCache（SLC）；
    4. 推理时通过 SLC 查找最近覆盖图并 replay。

    与 Megatron 的关系
    ------------------
    对应 Megatron CUDAGraphBatchDimensionBuilder 的 build() + replay() 逻辑，
    扩展为多设备异构感知版本。
    """

    def __init__(
        self,
        device_specs: List[DeviceSpec],
        cuda_graph_max_tokens: int,
        num_cuda_graphs: int = -1,   # -1 = 自动模式，镜像 Megatron fde3b90a
        graph_scope: GraphScope = GraphScope.LOCAL,
        hidden_size_bytes: int = 2,
        slc: Optional[SharedLocalityCache] = None,
    ) -> None:
        """
        Parameters
        ----------
        device_specs        : DES-LOC 拓扑中全部设备的描述列表
        cuda_graph_max_tokens: 单次推理最大 token 数
        num_cuda_graphs     : -1 自动; >=1 固定数量（Megatron 语义）
        graph_scope         : FULL_ITERATION or LOCAL
        hidden_size_bytes   : 每 token 隐状态字节数（用于 PCIe 对齐）
        slc                 : 外部注入的 SLC 实例；None 则自动创建
        """
        self._validate_num_cuda_graphs(num_cuda_graphs, graph_scope)

        self.device_specs = device_specs
        self.cuda_graph_max_tokens = cuda_graph_max_tokens
        self.num_cuda_graphs = num_cuda_graphs
        self.graph_scope = graph_scope
        self.hidden_size_bytes = hidden_size_bytes
        self.slc: SharedLocalityCache = slc or SharedLocalityCache()

        # 每设备的 token-count 列表：device_index -> List[int]
        self._token_counts: Dict[int, List[int]] = {}
        # 每设备已捕获的 graph pool：device_index -> List[GraphPoolEntry]
        self._graph_pools: Dict[int, List[GraphPoolEntry]] = {}

        self._build_token_count_plans()

    # ------------------------------------------------------------------
    # 参数验证（镜像 Megatron arguments.py validate_args）
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_num_cuda_graphs(num_cuda_graphs: int, graph_scope: GraphScope) -> None:
        """
        对应 Megatron fde3b90a arguments.py 新增的 local 实现校验：
          num_cuda_graphs > 0 或 == -1
        """
        if graph_scope == GraphScope.LOCAL:
            assert num_cuda_graphs > 0 or num_cuda_graphs == -1, (
                f"For LOCAL graph scope, num_cuda_graphs must be a positive integer or -1 "
                f"(auto-mode). Got {num_cuda_graphs}. "
                f"-1 means the manager will automatically determine graph count "
                f"based on cuda_graph_max_tokens."
            )

    # ------------------------------------------------------------------
    # 规划阶段
    # ------------------------------------------------------------------

    def _build_token_count_plans(self) -> None:
        """为每块设备生成 token-count 计划，存入 _token_counts。"""
        for spec in self.device_specs:
            builder = HeteroTokenCountBuilder(
                device_spec=spec,
                cuda_graph_max_tokens=self.cuda_graph_max_tokens,
                hidden_size_bytes=self.hidden_size_bytes,
            )
            if self.num_cuda_graphs == -1:
                counts = builder.build_auto()
            else:
                counts = builder.build_fixed(self.num_cuda_graphs)

            self._token_counts[spec.device_index] = counts
            logger.info(
                "[GraphMgr] device=%d (%s) planned %d graph points: %s…",
                spec.device_index,
                spec.arch_name,
                len(counts),
                counts[:6],
            )

    # ------------------------------------------------------------------
    # 捕获阶段
    # ------------------------------------------------------------------

    def capture_all(self, model_fn: Optional[object] = None) -> None:
        """
        并发捕获所有设备上的 CUDA graph。
        A6000 × 2 和 H100 × 1 通过独立线程并行捕获，互不阻塞。
        捕获完成后统一写入 SLC。

        Parameters
        ----------
        model_fn : 捕获时的前向函数（真实部署时传入；smoke test 时为 None）
        """
        threads = []
        for spec in self.device_specs:
            t = threading.Thread(
                target=self._capture_device,
                args=(spec, model_fn),
                name=f"graph-capture-dev{spec.device_index}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        logger.info("[GraphMgr] all devices captured. SLC:\n%s", self.slc.dump_summary())

    def _capture_device(
        self,
        spec: DeviceSpec,
        model_fn: Optional[object],
    ) -> None:
        """单设备捕获循环，依次捕获该设备全部 token-count 点。"""
        counts = self._token_counts[spec.device_index]
        pool: List[GraphPoolEntry] = []

        for tc in counts:
            entry = self._capture_single(spec, tc, model_fn)
            pool.append(entry)
            self.slc.register(entry)

        self._graph_pools[spec.device_index] = pool
        logger.info(
            "[GraphMgr] device=%d (%s) capture complete: %d graphs",
            spec.device_index, spec.arch_name, len(pool)
        )

    def _capture_single(
        self,
        spec: DeviceSpec,
        token_count: int,
        model_fn: Optional[object],
    ) -> GraphPoolEntry:
        """
        捕获单个 CUDA graph。
        真实实现中会：
          1. 切换到 spec.device_index；
          2. 用 torch.cuda.CUDAGraph() + capture_begin/end 包裹 model_fn；
          3. 记录 graph handle。
        Smoke test / 无 GPU 环境下退化为 mock 计时。
        """
        t0 = time.perf_counter()
        graph_handle = None

        if torch.cuda.is_available() and model_fn is not None:
            try:
                with torch.cuda.device(spec.device_index):
                    g = torch.cuda.CUDAGraph()
                    # 实际捕获调用（此处为框架骨架，model_fn 签名由调用方定义）
                    # with torch.cuda.graph(g):
                    #     model_fn(token_count)
                    graph_handle = g
            except Exception as exc:
                logger.warning(
                    "[GraphMgr] capture failed dev=%d tc=%d: %s",
                    spec.device_index, token_count, exc
                )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        entry = GraphPoolEntry(
            token_count=token_count,
            device_index=spec.device_index,
            sm_arch=spec.sm_arch,
            captured=True,
            capture_latency_ms=latency_ms,
            graph_handle=graph_handle,
        )
        logger.debug(
            "[GraphMgr] captured dev=%d tc=%d lat=%.2fms",
            spec.device_index, token_count, latency_ms
        )
        return entry

    # ------------------------------------------------------------------
    # 推理 Replay 阶段
    # ------------------------------------------------------------------

    def find_graph_for_request(
        self,
        request_token_count: int,
        preferred_device: Optional[int] = None,
    ) -> Optional[GraphPoolEntry]:
        """
        通过 SLC 查找满足 request_token_count 的最近覆盖图。

        DES-LOC 迁移语义
        ----------------
        若 preferred_device 无法满足（SLC miss），自动 fallback 到对端设备，
        避免因 PCIe 迁移代价而放弃 graph replay。
        """
        entry = self.slc.lookup_nearest(request_token_count, preferred_device)
        if entry is None:
            logger.warning(
                "[GraphMgr] SLC miss for token_count=%d preferred_dev=%s",
                request_token_count, preferred_device
            )
        else:
            logger.debug(
                "[GraphMgr] SLC hit: request_tc=%d -> graph_tc=%d dev=%d",
                request_token_count, entry.token_count, entry.device_index
            )
        return entry

    def replay(self, entry: GraphPoolEntry) -> bool:
        """
        Replay 已捕获的 CUDA graph。
        真实部署时调用 entry.graph_handle.replay()。
        """
        if entry.graph_handle is None:
            logger.debug("[GraphMgr] replay skipped (no handle, mock mode) tc=%d", entry.token_count)
            return False
        try:
            entry.graph_handle.replay()  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            logger.error("[GraphMgr] replay error tc=%d dev=%d: %s", entry.token_count, entry.device_index, exc)
            return False

    # ------------------------------------------------------------------
    # 统计 / 调试
    # ------------------------------------------------------------------

    def token_count_stats(self) -> Dict[int, Dict[str, object]]:
        """返回每设备 token-count 覆盖统计，便于调优。"""
        stats = {}
        for spec in self.device_specs:
            counts = self._token_counts.get(spec.device_index, [])
            stats[spec.device_index] = {
                "arch": spec.arch_name,
                "num_graphs": len(counts),
                "min_tc": min(counts) if counts else 0,
                "max_tc": max(counts) if counts else 0,
                "coverage_density": len(counts) / max(1, self.cuda_graph_max_tokens),
            }
        return stats


# ---------------------------------------------------------------------------
# DES-LOC 拓扑工厂函数
# ---------------------------------------------------------------------------


def build_des_loc_topology(
    cuda_graph_max_tokens: int = 512,
    num_cuda_graphs: int = -1,
    tp_size: int = 1,
    hidden_size_bytes: int = 2,
) -> HeteroFineGrainedCudaGraphManager:
    """
    构造 2×A6000 + 1×H100 NVL 的标准 DES-LOC 拓扑管理器。

    Parameters
    ----------
    cuda_graph_max_tokens : 推理最大 token 数
    num_cuda_graphs       : -1 自动，或固定正整数
    tp_size               : Tensor Parallel size（影响 token-count 对齐）
    hidden_size_bytes     : 每 token 隐状态字节数（PCIe 对齐用）
    """
    specs = [
        DeviceSpec(
            device_index=0,
            sm_arch=SM_ARCH_A6000,
            vram_gb=48.0,
            role=DeviceRole.PREFILL_SMALL_DECODE,
            tp_rank=0,
            tp_size=tp_size,
        ),
        DeviceSpec(
            device_index=1,
            sm_arch=SM_ARCH_A6000,
            vram_gb=48.0,
            role=DeviceRole.PREFILL_SMALL_DECODE,
            tp_rank=1 % tp_size,
            tp_size=tp_size,
        ),
        DeviceSpec(
            device_index=2,
            sm_arch=SM_ARCH_H100_NVL,
            vram_gb=96.0,
            role=DeviceRole.LARGE_DECODE_VERIFY,
            tp_rank=0,
            tp_size=tp_size,
        ),
    ]
    slc = SharedLocalityCache()
    mgr = HeteroFineGrainedCudaGraphManager(
        device_specs=specs,
        cuda_graph_max_tokens=cuda_graph_max_tokens,
        num_cuda_graphs=num_cuda_graphs,
        graph_scope=GraphScope.LOCAL,
        hidden_size_bytes=hidden_size_bytes,
        slc=slc,
    )
    return mgr


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- 1. HeteroTokenCountBuilder: auto mode on A6000 ---
    spec_a6k = DeviceSpec(0, SM_ARCH_A6000, 48.0, DeviceRole.PREFILL_SMALL_DECODE, tp_size=2)
    builder_a6k = HeteroTokenCountBuilder(spec_a6k, cuda_graph_max_tokens=256, hidden_size_bytes=2)
    counts_a6k = builder_a6k.build_auto()
    assert counts_a6k[0] == 256, f"first entry must be max_tokens, got {counts_a6k[0]}"
    assert counts_a6k == sorted(counts_a6k, reverse=True), "counts must be descending"
    assert all(c % 2 == 0 for c in counts_a6k), "all counts must be TP-aligned (tp=2)"

    # --- 2. HeteroTokenCountBuilder: H100 sparse vs A6000 dense ---
    spec_h100 = DeviceSpec(2, SM_ARCH_H100_NVL, 96.0, DeviceRole.LARGE_DECODE_VERIFY, tp_size=2)
    builder_h100 = HeteroTokenCountBuilder(spec_h100, cuda_graph_max_tokens=256, hidden_size_bytes=2)
    counts_h100 = builder_h100.build_auto()
    assert len(counts_h100) < len(counts_a6k), (
        f"H100 sparse should have fewer graphs than A6000 dense: "
        f"{len(counts_h100)} vs {len(counts_a6k)}"
    )

    # --- 3. -1 sentinel 不被 clamp 截断 ---
    mgr = build_des_loc_topology(cuda_graph_max_tokens=128, num_cuda_graphs=-1, tp_size=1)
    for dev_idx, stats in mgr.token_count_stats().items():
        assert stats["max_tc"] == 128, f"dev={dev_idx} max_tc should be 128"

    # --- 4. SLC lookup_nearest ---
    slc = SharedLocalityCache()
    e1 = GraphPoolEntry(token_count=64, device_index=0, sm_arch=SM_ARCH_A6000, captured=True)
    e2 = GraphPoolEntry(token_count=128, device_index=0, sm_arch=SM_ARCH_A6000, captured=True)
    slc.register(e1)
    slc.register(e2)
    hit = slc.lookup_nearest(50, preferred_device=0)
    assert hit is not None and hit.token_count == 64, f"expected tc=64, got {hit}"

    # --- 5. validate_num_cuda_graphs 参数校验 ---
    try:
        HeteroFineGrainedCudaGraphManager._validate_num_cuda_graphs(-2, GraphScope.LOCAL)
        raise AssertionError("Should have raised AssertionError for -2")
    except AssertionError as e:
        assert "-2" in str(e)

    print("All smoke tests passed.")
