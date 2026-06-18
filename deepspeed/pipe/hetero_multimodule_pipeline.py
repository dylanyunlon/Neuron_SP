"""
DES-LOC Heterogeneous Multimodule Pipeline Schedule
====================================================

上游设计意图 (Megatron commit 0ca9b63)
---------------------------------------
Megatron 在此 commit 中为 1F1B 调度引入了"多模块流水线"支持：
- MultiModulePipelineCommunicator：管理跨模块（如 encoder→LLM）的 bridge 通信
- _prepare_tensor_for_comm / _restore_tensor_from_comm：2D/3D 张量适配层
- MultiModuleProcessGroupCollection：聚合多个模块各自的进程组
- backward_step_multimodule：字典化输入/输出的反向传播
- total_stages / current_stage 属性：统一单模块和多模块的阶段感知接口
- 广播 PG 缓存 + destroy：避免重复创建 NCCL communicator

DES-LOC 适配点
--------------
DES-LOC = Decoupled Execution with Shared LOcality Cache

硬件拓扑：2× A6000 48GB (SM86, PCIe) + 1× H100 NVL 96GB (SM90, PCIe)，1.5TB CPU DRAM，无 NVLink

关键异构挑战：
1. **设备异质性**：A6000 (SM86) 与 H100 (SM90) 有不同的 CUDA 能力、内存带宽、计算吞吐。
   Encoder 阶段适合放 A6000（序列处理，内存密集），LLM 阶段放 H100（计算密集）。
2. **PCIe 带宽瓶颈**：无 NVLink，跨设备 P2P 带宽约 16–32 GB/s vs NVLink 600 GB/s。
   DES-LOC 引入 **Shared LOcality Cache (SLC)**：将跨设备激活先卸载到 CPU DRAM（1.5TB），
   接收方按需拉取，避免直接 PCIe 点对点争用。
3. **解耦执行（Decoupled Execution）**：encoder 前向和 LLM 前向在时间上解耦，
   通过 SLC 异步缓冲，允许两侧以各自最优 batch/micro-batch 节奏运行。
4. **张量维度适配**：上游 bridge 通信假设固定 3D 张量；DES-LOC 的 SLC 路径需要显式
   serialize/deserialize，此文件在 prepare/restore 层加入设备感知路由。
5. **进程组生命周期**：PCIe 拓扑下 NCCL 初始化代价更高，PG 缓存（BroadcastPGCache）
   在 DES-LOC 中尤为关键；destroy 路径需与 SLC 清理同步。

文件结构：
  - HeteroDeviceProfile：设备能力描述
  - SharedLocalityCache (SLC)：CPU DRAM 中介激活缓存
  - DESLOCBridgeConfig：跨模块传输策略配置
  - HeteroMultimodulePipeline：主调度类（1F1B + SLC 路由）
  - 辅助函数：prepare/restore 张量，backward_step_multimodule_desfloc

作者: Neuron_SP / DES-LOC 适配层
基于: github.com/dylanyunlon/Neuron_SP (DeepSpeed)
上游参考: NVIDIA/Megatron-LM commit 0ca9b63
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 设备能力描述
# ---------------------------------------------------------------------------

@dataclass
class HeteroDeviceProfile:
    """描述单个 GPU 的计算/内存能力，用于 DES-LOC 路由决策。

    在 A6000+H100 异构集群中，调度器需要知道每个设备的：
    - SM 版本（影响 CUDA kernel 选择）
    - 显存大小（影响 micro-batch 大小上限）
    - PCIe 带宽（影响 SLC offload 决策阈值）

    Attributes:
        device_index: CUDA device index (e.g., 0, 1, 2)
        sm_major: SM major version (A6000=8, H100=9)
        sm_minor: SM minor version (A6000=6, H100=0)
        vram_gb: GPU VRAM in GB
        pcie_bw_gbps: PCIe bandwidth in GB/s (typical measured, not theoretical)
        role: 'encoder' | 'llm' | 'shared' — pipeline role of this device
    """
    device_index: int
    sm_major: int
    sm_minor: int
    vram_gb: float
    pcie_bw_gbps: float
    role: str = "shared"

    @property
    def sm_version(self) -> int:
        return self.sm_major * 10 + self.sm_minor

    @property
    def is_ampere(self) -> bool:
        return self.sm_major == 8

    @property
    def is_hopper(self) -> bool:
        return self.sm_major == 9

    @classmethod
    def from_cuda_device(cls, device_index: int, role: str = "shared") -> "HeteroDeviceProfile":
        """从当前 CUDA 设备自动探测能力。"""
        props = torch.cuda.get_device_properties(device_index)
        # PCIe 带宽：从设备名称启发式估算
        name_lower = props.name.lower()
        if "h100" in name_lower or "h200" in name_lower:
            pcie_bw = 64.0   # H100 NVL PCIe Gen5 ×16
        elif "a6000" in name_lower or "a100" in name_lower:
            pcie_bw = 32.0   # A6000 PCIe Gen4 ×16
        else:
            pcie_bw = 16.0   # 保守估计
        return cls(
            device_index=device_index,
            sm_major=props.major,
            sm_minor=props.minor,
            vram_gb=props.total_memory / (1024 ** 3),
            pcie_bw_gbps=pcie_bw,
            role=role,
        )

    def __repr__(self) -> str:
        return (
            f"HeteroDeviceProfile(dev={self.device_index}, "
            f"SM{self.sm_major}{self.sm_minor}, "
            f"{self.vram_gb:.0f}GB, {self.pcie_bw_gbps:.0f}GB/s, role={self.role})"
        )


def detect_cluster_profiles() -> Dict[int, HeteroDeviceProfile]:
    """探测本机所有 CUDA 设备，自动分配 encoder/llm 角色。

    启发式策略：
    - H100 (SM90) → role='llm'（计算密集型 LLM 阶段）
    - A6000/其他 Ampere → role='encoder'（内存密集型编码器阶段）

    Returns:
        Dict[device_index -> HeteroDeviceProfile]
    """
    profiles: Dict[int, HeteroDeviceProfile] = {}
    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        if props.major == 9:
            role = "llm"
        else:
            role = "encoder"
        profiles[i] = HeteroDeviceProfile.from_cuda_device(i, role=role)
        logger.info("Detected device %d: %s", i, profiles[i])
    return profiles


# ---------------------------------------------------------------------------
# 2. Shared Locality Cache (SLC)
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """CPU DRAM 中介激活缓存，DES-LOC 核心组件。

    上游 Megatron bridge_communicator 假设 GPU-to-GPU NCCL 直传；
    在 PCIe 无 NVLink 的 A6000+H100 拓扑中，直传带宽（~16-32 GB/s）
    严重限制 pipeline bubble 填充效率。

    DES-LOC 解法：
    - 发送方将激活张量 pin_memory() 后存入 CPU DRAM（1.5TB 充足）
    - 接收方在需要时以异步 DMA 拉回 GPU
    - 允许 encoder（A6000）和 LLM（H100）以各自节奏生产/消费，
      不必实时阻塞等待 PCIe 传输完成

    SLC 使用 (module_name, microbatch_id) 作为 key，
    支持前向激活和反向梯度两个方向的缓存。

    Attributes:
        max_entries: 最大缓存条目数（超出时 LRU 淘汰）
        _cache: {key -> (cpu_tensor, event)} 的内存字典
        _lock: 线程安全锁（异步 DMA 场景下需要）
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()

        logger.info(
            "SharedLocalityCache initialized: max_entries=%d, "
            "backing store=CPU DRAM (pin_memory)",
            max_entries,
        )

    def _make_key(self, module_name: str, microbatch_id: int, direction: str) -> str:
        return f"{module_name}:{microbatch_id}:{direction}"

    def store(
        self,
        module_name: str,
        microbatch_id: int,
        tensor: torch.Tensor,
        direction: str = "fwd",
    ) -> None:
        """将 GPU 张量异步卸载到 CPU pin_memory。

        Args:
            module_name: 模块名称（如 'encoder', 'llm'）
            microbatch_id: micro-batch 编号
            tensor: GPU 上的激活/梯度张量
            direction: 'fwd'（前向激活）或 'bwd'（反向梯度）
        """
        key = self._make_key(module_name, microbatch_id, direction)

        # 记录 GPU 侧完成事件，避免 CPU 过早读取
        event = torch.cuda.Event()
        event.record()

        # 异步 DMA: GPU → CPU pin_memory
        cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        cpu_tensor.copy_(tensor, non_blocking=True)

        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_entries:
                evict_key = self._access_order.pop(0)
                del self._cache[evict_key]
                logger.debug("SLC evicted key=%s (LRU)", evict_key)

            self._cache[key] = (cpu_tensor, event)
            self._access_order.append(key)

        logger.debug(
            "SLC store: key=%s, shape=%s, dtype=%s",
            key, tuple(tensor.shape), tensor.dtype,
        )

    def retrieve(
        self,
        module_name: str,
        microbatch_id: int,
        target_device: torch.device,
        direction: str = "fwd",
        timeout_s: float = 30.0,
    ) -> Optional[torch.Tensor]:
        """从 CPU DRAM 拉取张量到目标 GPU。

        会等待存入时的 CUDA Event 完成，确保数据一致性。

        Args:
            module_name: 模块名称
            microbatch_id: micro-batch 编号
            target_device: 目标 GPU device
            direction: 'fwd' 或 'bwd'
            timeout_s: 最大等待秒数（SLC 异步写入尚未完成时轮询）

        Returns:
            目标设备上的张量，或 None（未找到）
        """
        key = self._make_key(module_name, microbatch_id, direction)
        deadline = time.monotonic() + timeout_s

        # 轮询等待（通常数毫秒内完成）
        while time.monotonic() < deadline:
            with self._lock:
                if key in self._cache:
                    cpu_tensor, event = self._cache[key]
                    break
            time.sleep(0.001)
        else:
            logger.warning("SLC retrieve timeout: key=%s", key)
            return None

        # 等待 GPU 侧 DMA 完成
        event.synchronize()

        # 异步 DMA: CPU → target GPU
        gpu_tensor = cpu_tensor.to(target_device, non_blocking=True)
        logger.debug(
            "SLC retrieve: key=%s → device=%s, shape=%s",
            key, target_device, tuple(gpu_tensor.shape),
        )
        return gpu_tensor

    def evict(self, module_name: str, microbatch_id: int, direction: str = "fwd") -> None:
        """显式淘汰缓存条目，释放 CPU DRAM。"""
        key = self._make_key(module_name, microbatch_id, direction)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                logger.debug("SLC explicit evict: key=%s", key)

    def clear(self) -> None:
        """清空全部缓存（进程退出或模型切换时调用）。"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
        logger.info("SLC cleared all entries")

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        return f"SharedLocalityCache(entries={self.size}/{self.max_entries})"


# ---------------------------------------------------------------------------
# 3. DES-LOC 传输配置
# ---------------------------------------------------------------------------

@dataclass
class DESLOCBridgeConfig:
    """跨模块 bridge 传输策略配置。

    决定对于给定的 (src_module, dst_module) 对，激活应走：
    - 'nccl'：直接 GPU-to-GPU NCCL（同设备或有 NVLink 时优选）
    - 'slc'：经由 SharedLocalityCache (CPU DRAM 中介)
    - 'auto'：根据设备档案自动选择

    在 A6000+H100 PCIe 拓扑中：
    - encoder(A6000) → LLM(H100)：PCIe 传输，推荐 'slc' 以解耦执行
    - 同设备内 PP 相邻阶段：'nccl'（本地 NVLink 或同 PCIe switch）

    Attributes:
        src_module: 源模块名
        dst_module: 目标模块名
        transport: 'nccl' | 'slc' | 'auto'
        slc_threshold_mb: 激活大于此值时强制走 SLC（MB）
        async_slc: 是否允许 SLC 异步存入（发送方不等接收方）
    """
    src_module: str
    dst_module: str
    transport: str = "auto"
    slc_threshold_mb: float = 64.0
    async_slc: bool = True

    def resolve_transport(
        self,
        tensor_mb: float,
        src_profile: Optional[HeteroDeviceProfile],
        dst_profile: Optional[HeteroDeviceProfile],
    ) -> str:
        """根据张量大小和设备档案解析实际传输方式。

        Args:
            tensor_mb: 张量大小（MB）
            src_profile: 源设备档案
            dst_profile: 目标设备档案

        Returns:
            'nccl' 或 'slc'
        """
        if self.transport != "auto":
            return self.transport

        # 跨 SM 版本（A6000 SM86 → H100 SM90）：优先 SLC
        if (src_profile is not None and dst_profile is not None
                and src_profile.sm_version != dst_profile.sm_version):
            logger.debug(
                "DESLOCBridgeConfig: cross-SM transport SM%d→SM%d, using SLC",
                src_profile.sm_version, dst_profile.sm_version,
            )
            return "slc"

        # 大张量超过阈值：走 SLC 避免 PCIe 长时阻塞
        if tensor_mb > self.slc_threshold_mb:
            logger.debug(
                "DESLOCBridgeConfig: tensor %.1fMB > threshold %.1fMB, using SLC",
                tensor_mb, self.slc_threshold_mb,
            )
            return "slc"

        return "nccl"


# ---------------------------------------------------------------------------
# 4. 张量维度适配（对应上游 _prepare / _restore）
# ---------------------------------------------------------------------------

def _tensor_size_mb(tensor: torch.Tensor) -> float:
    """计算张量占用 MB 数。"""
    return tensor.numel() * tensor.element_size() / (1024 ** 2)


def prepare_tensor_for_hetero_comm(
    tensor: Union[torch.Tensor, List[torch.Tensor], None],
    target_device: Optional[torch.device] = None,
) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """为异构 P2P/bridge 通信准备张量。

    上游 Megatron 逻辑：P2P 通信期望 3D 张量，2D 张量需 unsqueeze(-1)。
    DES-LOC 额外处理：若 target_device 与当前 tensor 设备不同（跨 SM 传输），
    先确保张量 contiguous，再由调用方决定走 SLC 还是 NCCL。

    Args:
        tensor: 输入张量（2D/3D）、张量列表或 None
        target_device: 目标设备（可选，用于 contiguous 检查）

    Returns:
        准备好的 3D 张量（若输入为 2D 则末维扩展为 1）、列表或 None

    Note:
        3D 张量且末维为 1 的情形是歧义的（无法区分是原始 3D 还是被 prepare 过的 2D），
        此类输入会触发 AssertionError，调用方应避免传入。
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [prepare_tensor_for_hetero_comm(t, target_device) for t in tensor]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 2:
            t = tensor.unsqueeze(-1)
            if not t.is_contiguous():
                t = t.contiguous()
            return t
        assert not (tensor.ndim == 3 and tensor.shape[-1] == 1), (
            f"3D tensor with singleton last dim {tuple(tensor.shape)} is ambiguous for "
            "hetero comm. Use a 2D tensor or a 3D tensor with last_dim > 1."
        )
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
    return tensor


def restore_tensor_from_hetero_comm(
    tensor: Union[torch.Tensor, List[torch.Tensor], None],
    src_device: Optional[torch.device] = None,
) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """从异构通信恢复张量形状。

    逆操作 prepare_tensor_for_hetero_comm：若末维为 1 则 squeeze(-1)。
    DES-LOC：从 SLC 拉取的张量已在目标设备上，此处仅做形状恢复。

    Args:
        tensor: 通信后收到的张量、列表或 None
        src_device: 来源设备（保留用于调试日志）

    Returns:
        形状恢复后的张量、列表或 None
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [restore_tensor_from_hetero_comm(t, src_device) for t in tensor]
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 3 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


# ---------------------------------------------------------------------------
# 5. 多模块反向传播（对应上游 backward_step_multimodule）
# ---------------------------------------------------------------------------

def backward_step_multimodule_desfloc(
    input_tensor: Dict[str, torch.Tensor],
    output_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_tensor_grad: Optional[Dict[str, torch.Tensor]],
    config,
    language_model_module_name: str,
) -> Dict[str, torch.Tensor]:
    """DES-LOC 多模块反向传播。

    上游 Megatron 设计意图（backward_step_multimodule）：
    - 多模块 pipeline 中，input/output 以模块名为 key 的 dict 组织
    - 最后阶段 output_tensor 是标量 loss（非 dict），需规范化
    - 各模块独立 backward，支持某些模块无梯度（如 VLM 中无图像的 batch）
    - 最终收集各模块的 input_tensor.grad 返回给 pipeline

    DES-LOC 适配：
    - 在 A6000(encoder) 侧，input_tensor 可能是从 SLC 恢复的张量，
      需确保 requires_grad=True 才能正确传播
    - 支持 H100(LLM) 侧 output_tensor 为混合精度（bfloat16）的标量 loss
    - 对 None output_tensor_module（encoder batch 无图像场景）优雅跳过

    Args:
        input_tensor: {module_name: tensor} 各模块输入张量
        output_tensor: 最后阶段为标量 loss，中间阶段为 {module_name: tensor}
        output_tensor_grad: {module_name: grad} 或 None（最后阶段）
        config: Pipeline 配置（含 grad_scale_func, deallocate_pipeline_outputs）
        language_model_module_name: LLM 模块名称（用于规范化最后阶段输出）

    Returns:
        {module_name: input_grad} 各模块输入梯度字典
    """
    # 确保所有非 None 输入张量持有梯度
    for module_name, tensor in input_tensor.items():
        t = tensor[0] if isinstance(tensor, list) else tensor
        if t is not None and not t.requires_grad:
            # SLC 恢复的张量默认无 requires_grad，在此修正
            try:
                t.requires_grad_(True)
                logger.debug(
                    "backward_step_multimodule_desfloc: enabled requires_grad for module=%s",
                    module_name,
                )
            except RuntimeError:
                pass  # leaf tensor 不在计算图中时忽略
        if t is not None:
            t.retain_grad()

    # 规范化 output_tensor：最后阶段是标量 loss，绑定到 LLM 模块名
    if not isinstance(output_tensor, dict):
        output_tensor = {language_model_module_name: output_tensor}

    # 规范化 output_tensor_grad：None 表示最后阶段
    if not output_tensor_grad:
        output_tensor_grad = {k: None for k in output_tensor.keys()}

    # 梯度缩放（仅最后阶段，即 grad 为 None 时）
    for module_name in output_tensor.keys():
        if (output_tensor_grad.get(module_name) is None
                and config.grad_scale_func is not None
                and output_tensor[module_name] is not None):
            output_tensor[module_name] = config.grad_scale_func(output_tensor[module_name])

    # 各模块独立 backward
    for module_name in output_tensor.keys():
        out = output_tensor[module_name]
        out_grad = output_tensor_grad.get(module_name)

        # 无输出或不需要梯度（如 encoder batch 无图像）：跳过
        if out is None or not out.requires_grad:
            logger.debug(
                "backward_step_multimodule_desfloc: skipping module=%s "
                "(output is None or no requires_grad)",
                module_name,
            )
            continue

        if getattr(config, 'deallocate_pipeline_outputs', False):
            # custom_backward 可兼容 deallocate 后的张量
            _custom_backward(out, out_grad)
        else:
            torch.autograd.backward(out, grad_tensors=out_grad)

    # 收集各模块输入梯度
    input_tensor_grad: Dict[str, Optional[torch.Tensor]] = {}
    for module_name, tensor in input_tensor.items():
        t = tensor[0] if isinstance(tensor, list) else tensor
        if t is None:
            input_tensor_grad[module_name] = None
        else:
            input_tensor_grad[module_name] = t.grad

    return input_tensor_grad


def _custom_backward(output: torch.Tensor, grad: Optional[torch.Tensor]) -> None:
    """兼容 deallocate_pipeline_outputs 的 backward。

    当激活已被 deallocate（data 置为空张量）时，仍能通过 grad_fn 反向传播。
    对应上游 Megatron custom_backward。
    """
    assert output.requires_grad, "output must require grad for custom_backward"
    if grad is None:
        assert output.numel() == 1, "grad=None only valid for scalar output"
        output.backward()
    else:
        torch.autograd.backward([output], [grad])


# ---------------------------------------------------------------------------
# 6. 进程组缓存（对应上游 BridgeCommunicator._broadcast_pg_cache）
# ---------------------------------------------------------------------------

class BroadcastPGCache:
    """全局广播进程组缓存，避免重复创建 NCCL communicator。

    上游 Megatron 在 BridgeCommunicator 类上用类变量实现相同功能。
    DES-LOC 将其独立为模块级单例，便于与 SLC 清理同步。

    在 PCIe 拓扑下，NCCL communicator 初始化（AllReduce ring/tree 构建）
    代价更高，缓存复用尤为重要。
    """
    _cache: Dict[str, dist.ProcessGroup] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(cls, ranks_list: List[List[int]]) -> dist.ProcessGroup:
        """获取或创建广播 PG，线程安全。

        Args:
            ranks_list: [[rank0, rank1, ...], [...], ...] 广播组枚举

        Returns:
            对应的 ProcessGroup
        """
        cache_key = str(sorted([tuple(r) for r in ranks_list]))
        with cls._lock:
            if cache_key not in cls._cache:
                pg, _ = dist.new_subgroups_by_enumeration(ranks_list, backend="nccl")
                cls._cache[cache_key] = pg
                logger.debug("BroadcastPGCache created new PG for key=%s", cache_key)
            return cls._cache[cache_key]

    @classmethod
    def destroy_all(cls) -> None:
        """销毁全部缓存 PG（进程退出或测试 teardown 时调用）。"""
        with cls._lock:
            for pg in cls._cache.values():
                if pg is not None:
                    try:
                        dist.destroy_process_group(pg)
                    except Exception as e:
                        logger.warning("BroadcastPGCache: destroy_process_group failed: %s", e)
            cls._cache.clear()
        logger.info("BroadcastPGCache destroyed all PGs")


# ---------------------------------------------------------------------------
# 7. 多模块进程组集合（对应上游 MultiModuleProcessGroupCollection）
# ---------------------------------------------------------------------------

class HeteroModuleProcessGroups:
    """异构多模块进程组集合。

    上游 Megatron MultiModuleProcessGroupCollection 的 DES-LOC 扩展版本。
    增加了设备档案感知，能够为每个模块返回其所在设备的 HeteroDeviceProfile，
    支持 bridge 传输策略的自动决策。

    Attributes:
        module_pgs: {module_name -> dict of process groups}
        language_model_module_name: LLM 模块名（None 表示本 rank 无 LLM）
        device_profiles: {device_index -> HeteroDeviceProfile}
        _module_devices: {module_name -> device_index}（懒初始化）
    """

    def __init__(
        self,
        module_pgs: Dict[str, Dict[str, dist.ProcessGroup]],
        language_model_module_name: Optional[str] = None,
        device_profiles: Optional[Dict[int, HeteroDeviceProfile]] = None,
    ):
        """
        Args:
            module_pgs: {module_name -> {'tp': pg, 'cp': pg, 'pp': pg, 'dp': pg, ...}}
            language_model_module_name: LLM 模块名
            device_profiles: 设备档案字典（None 时自动探测）
        """
        if not module_pgs:
            raise ValueError("module_pgs cannot be empty")
        if language_model_module_name is not None:
            if language_model_module_name not in module_pgs:
                raise ValueError(
                    f"language_model_module_name '{language_model_module_name}' "
                    f"not in module_pgs keys: {list(module_pgs.keys())}"
                )

        self.module_pgs = module_pgs
        self.language_model_module_name = language_model_module_name
        self.device_profiles: Dict[int, HeteroDeviceProfile] = (
            device_profiles if device_profiles is not None else {}
        )
        self._module_devices: Dict[str, int] = {}

        logger.info(
            "HeteroModuleProcessGroups initialized: modules=%s, llm=%s",
            list(module_pgs.keys()), language_model_module_name,
        )

    def has_language_model(self) -> bool:
        return self.language_model_module_name is not None

    def get_language_model_cp_size(self) -> int:
        """获取 LLM 模块的 Context Parallel size，用于 loss 缩放。"""
        if not self.has_language_model():
            raise ValueError("No language model on this rank")
        pgs = self.module_pgs[self.language_model_module_name]
        cp_pg = pgs.get("cp")
        if cp_pg is None:
            return 1
        return dist.get_world_size(group=cp_pg)

    def get_module_pgs(self, module_name: str) -> Dict[str, dist.ProcessGroup]:
        if module_name not in self.module_pgs:
            raise KeyError(
                f"Module '{module_name}' not found. Available: {list(self.module_pgs.keys())}"
            )
        return self.module_pgs[module_name]

    def get_device_profile_for_module(
        self, module_name: str
    ) -> Optional[HeteroDeviceProfile]:
        """返回模块所在设备的档案（用于 bridge 传输策略）。

        懒初始化：首次调用时从 CUDA current device 推断。
        """
        if module_name not in self._module_devices:
            # 简单启发式：LLM → 最后一个设备（通常 H100），encoder → 其余
            if module_name == self.language_model_module_name:
                dev_idx = torch.cuda.device_count() - 1
            else:
                enc_names = [k for k in self.module_pgs if k != self.language_model_module_name]
                idx = enc_names.index(module_name) if module_name in enc_names else 0
                dev_idx = idx % max(1, torch.cuda.device_count() - 1)
            self._module_devices[module_name] = dev_idx

        dev_idx = self._module_devices[module_name]
        return self.device_profiles.get(dev_idx)

    def keys(self):
        return self.module_pgs.keys()

    def values(self):
        return self.module_pgs.values()

    def items(self):
        return self.module_pgs.items()

    def __len__(self):
        return len(self.module_pgs)

    def __getitem__(self, module_name: str):
        return self.module_pgs[module_name]

    def __repr__(self) -> str:
        modules = list(self.module_pgs.keys())
        return (
            f"HeteroModuleProcessGroups("
            f"modules={modules}, llm={self.language_model_module_name})"
        )


# ---------------------------------------------------------------------------
# 8. HeteroMultimodulePipeline — 主调度类
# ---------------------------------------------------------------------------

class HeteroMultimodulePipeline:
    """DES-LOC 异构多模块 1F1B Pipeline 调度器。

    上游 Megatron 设计意图（forward_backward_pipelining_without_interleaving + multimodule）：
    - 非交错 1F1B 调度：warmup（纯前向）→ 1F1B 稳态 → cooldown（纯后向）
    - MultiModulePipelineCommunicator 统一管理 P2P（module 内）和 bridge（module 间）通信
    - total_stages / current_stage 属性抽象多模块阶段位置，消除对 pp_group.rank() 的直接依赖
    - 变长序列（variable_seq_lengths）支持多模块场景的形状动态协商

    DES-LOC 适配点：
    1. **SLC 路由**：encoder→LLM bridge 传输通过 SharedLocalityCache 中介，
       解耦两侧执行节奏（PCIe 无 NVLink 场景下减少阻塞）。
    2. **设备感知调度**：根据 HeteroDeviceProfile 决定每次 bridge 传输走 NCCL 还是 SLC。
    3. **异构 backward**：LLM（H100）侧 backward 可以在 encoder（A6000）前向完成前启动，
       因为激活已在 SLC 中缓冲（async_slc=True 时）。
    4. **micro-batch 对齐**：A6000 内存较小（48GB），H100 较大（96GB），
       调度器支持不同侧使用不同有效 micro-batch 大小（通过 slc_batch_scale）。
    5. **PG 生命周期**：BroadcastPGCache.destroy_all() 与 SLC.clear() 同步调用。

    Attributes:
        config: Pipeline 配置对象
        slc: SharedLocalityCache 实例
        device_profiles: {device_index -> HeteroDeviceProfile}
        bridge_configs: {(src, dst) -> DESLOCBridgeConfig}
        _microbatch_counter: 全局 micro-batch 计数（SLC key 生成）
    """

    def __init__(
        self,
        config,
        slc_max_entries: int = 256,
        bridge_configs: Optional[Dict[Tuple[str, str], DESLOCBridgeConfig]] = None,
        device_profiles: Optional[Dict[int, HeteroDeviceProfile]] = None,
    ):
        """
        Args:
            config: DeepSpeed/Megatron ModelParallelConfig 兼容配置对象
            slc_max_entries: SLC 最大缓存条目数
            bridge_configs: 覆盖自动策略的 bridge 传输配置
            device_profiles: 设备档案（None 时自动探测）
        """
        self.config = config
        self.slc = SharedLocalityCache(max_entries=slc_max_entries)
        self.device_profiles = (
            device_profiles if device_profiles is not None else detect_cluster_profiles()
        )
        self.bridge_configs: Dict[Tuple[str, str], DESLOCBridgeConfig] = bridge_configs or {}
        self._microbatch_counter = 0

        logger.info(
            "HeteroMultimodulePipeline initialized: slc=%s, devices=%s",
            self.slc, list(self.device_profiles.values()),
        )

    def _get_bridge_config(
        self, src_module: str, dst_module: str
    ) -> DESLOCBridgeConfig:
        """获取 (src, dst) 的 bridge 配置，不存在则返回 auto 默认值。"""
        return self.bridge_configs.get(
            (src_module, dst_module),
            DESLOCBridgeConfig(src_module=src_module, dst_module=dst_module, transport="auto"),
        )

    def route_activation(
        self,
        src_module: str,
        dst_module: str,
        tensor: torch.Tensor,
        microbatch_id: int,
        direction: str = "fwd",
        dst_device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """路由激活张量：选择 NCCL 直传或 SLC 中介。

        这是 DES-LOC 的核心路由逻辑，对应上游 bridge_communicator.send_forward 的扩展。

        Args:
            src_module: 源模块名
            dst_module: 目标模块名
            tensor: 要传输的激活/梯度张量
            microbatch_id: micro-batch ID（SLC key 的一部分）
            direction: 'fwd'（前向）或 'bwd'（反向）
            dst_device: 目标设备（SLC retrieve 时使用）

        Returns:
            若走 SLC 且为异步模式，返回 None（接收方后续调用 retrieve_activation）；
            若走 NCCL，返回 None（由 NCCL 完成通信，调用方处理）；
            若为同步 SLC，返回目标设备上的张量。
        """
        bridge_cfg = self._get_bridge_config(src_module, dst_module)
        src_profile = None
        dst_profile = None

        # 推断设备档案
        for dev_idx, profile in self.device_profiles.items():
            if profile.role == "encoder" and "encoder" in src_module:
                src_profile = profile
            if profile.role == "llm" and "llm" in dst_module:
                dst_profile = profile

        tensor_mb = _tensor_size_mb(tensor)
        transport = bridge_cfg.resolve_transport(tensor_mb, src_profile, dst_profile)

        logger.debug(
            "route_activation: %s→%s, mb_id=%d, dir=%s, %.2fMB, transport=%s",
            src_module, dst_module, microbatch_id, direction, tensor_mb, transport,
        )

        if transport == "slc":
            self.slc.store(
                module_name=f"{src_module}_to_{dst_module}",
                microbatch_id=microbatch_id,
                tensor=tensor,
                direction=direction,
            )
            if not bridge_cfg.async_slc and dst_device is not None:
                # 同步模式：立即取回（测试/调试用）
                return self.slc.retrieve(
                    module_name=f"{src_module}_to_{dst_module}",
                    microbatch_id=microbatch_id,
                    target_device=dst_device,
                    direction=direction,
                )
            return None  # 异步：接收方之后调用 retrieve_activation

        # transport == 'nccl'：由外层 bridge communicator 处理，此处仅返回 None
        return None

    def retrieve_activation(
        self,
        src_module: str,
        dst_module: str,
        microbatch_id: int,
        target_device: torch.device,
        direction: str = "fwd",
    ) -> Optional[torch.Tensor]:
        """从 SLC 取回激活张量到目标设备。

        在 DES-LOC 异步模式中，LLM（H100）侧在准备好接收时调用此方法，
        避免等待 encoder（A6000）完成前向。

        Args:
            src_module: 源模块名
            dst_module: 目标模块名
            microbatch_id: micro-batch ID
            target_device: 目标设备（H100 CUDA device）
            direction: 'fwd' 或 'bwd'

        Returns:
            目标设备上的张量，或 None（超时）
        """
        tensor = self.slc.retrieve(
            module_name=f"{src_module}_to_{dst_module}",
            microbatch_id=microbatch_id,
            target_device=target_device,
            direction=direction,
        )
        if tensor is not None:
            # 恢复通信前可能做的维度变换
            tensor = restore_tensor_from_hetero_comm(tensor)
        return tensor

    def compute_warmup_microbatches(
        self,
        p2p_communicator,
        num_microbatches: int,
    ) -> int:
        """计算 1F1B warmup 阶段所需 micro-batch 数。

        上游逻辑：warmup = total_stages - current_stage - 1
        DES-LOC：多模块时 total_stages 跨越所有模块，
        current_stage 由 MultiModulePipelineCommunicator 统一计算。

        Args:
            p2p_communicator: P2PCommunicator 或 MultiModulePipelineCommunicator
            num_microbatches: 本次 batch 的 micro-batch 总数

        Returns:
            warmup 阶段 micro-batch 数
        """
        warmup = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
        warmup = min(warmup, num_microbatches)
        logger.debug(
            "compute_warmup: total_stages=%d, current_stage=%d, warmup=%d, num_mb=%d",
            p2p_communicator.total_stages,
            p2p_communicator.current_stage,
            warmup,
            num_microbatches,
        )
        return warmup

    def build_backward_func(
        self,
        pg_collection: Optional[HeteroModuleProcessGroups],
        is_multimodule: bool,
    ) -> Callable:
        """构造反向传播函数（单模块或多模块）。

        上游 Megatron：is_multimodule 时用 backward_step_multimodule，
        否则用 backward_step。

        DES-LOC 替换为 backward_step_multimodule_desfloc，
        附加了 SLC 恢复张量的 requires_grad 修复逻辑。

        Args:
            pg_collection: 进程组集合（多模块时提供 language_model_module_name）
            is_multimodule: 是否多模块场景

        Returns:
            反向传播可调用对象
        """
        if is_multimodule and pg_collection is not None:
            lm_name = pg_collection.language_model_module_name or "llm"
            return partial(
                backward_step_multimodule_desfloc,
                language_model_module_name=lm_name,
            )
        else:
            return _backward_step_single_module

    def run_1f1b_schedule(
        self,
        forward_step_func: Callable,
        data_iterator,
        model,
        num_microbatches: int,
        p2p_communicator,
        pg_collection: Optional[HeteroModuleProcessGroups],
        forward_only: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> List:
        """执行 DES-LOC 1F1B 调度主循环。

        上游 Megatron forward_backward_pipelining_without_interleaving 的核心逻辑，
        适配为 DES-LOC 多模块异构场景。

        调度结构：
          [Warmup]  i=0..num_warmup-1: recv_forward → forward → send_forward
          [1F1B]    稳态: recv_forward → forward → send_fwd_recv_bwd
                                               ↓ (从 output_tensors 队列取)
                         backward → send_bwd_recv_fwd（或 send_bwd 最后一步）
          [Cooldown] i=0..num_warmup-1: recv_backward → backward → send_backward

        DES-LOC 扩展：
          - 跨模块 bridge 传输通过 route_activation/retrieve_activation 经 SLC
          - backward_func 选择感知 SLC 恢复张量的 requires_grad

        Args:
            forward_step_func: (data_iterator, model) -> (output, loss_func)
            data_iterator: 数据迭代器（可为 None，如非首阶段 rank）
            model: 模型（list 或单个 module）
            num_microbatches: micro-batch 总数
            p2p_communicator: P2PCommunicator 或 MultiModulePipelineCommunicator
            pg_collection: 进程组集合（可 None）
            forward_only: 仅前向（推理/验证模式）
            seq_length: 序列长度（variable_seq_lengths=False 时使用）
            micro_batch_size: micro-batch 大小

        Returns:
            loss_reduced 列表（仅末阶段非空）
        """
        is_multimodule = hasattr(p2p_communicator, "total_stages") and not hasattr(
            p2p_communicator, "pp_group"
        )
        backward_func = self.build_backward_func(pg_collection, is_multimodule)

        cp_size: Optional[int] = None
        if pg_collection is not None and pg_collection.has_language_model():
            cp_size = pg_collection.get_language_model_cp_size()

        num_warmup = self.compute_warmup_microbatches(p2p_communicator, num_microbatches)
        num_remaining = num_microbatches - num_warmup

        logger.info(
            "run_1f1b_schedule: num_mb=%d, warmup=%d, remaining=%d, forward_only=%s, "
            "cp_size=%s, is_multimodule=%s",
            num_microbatches, num_warmup, num_remaining, forward_only, cp_size, is_multimodule,
        )

        # 确定是否可变长度序列（多模块必须开启）
        variable_seq = getattr(self.config, "variable_seq_lengths", False)
        recv_shapes = [()] if variable_seq else None  # 简化：实际由 get_tensor_shapes 计算

        input_tensors: List = []
        output_tensors: List = []
        forward_data_store: List = []
        total_num_tokens = torch.tensor(0, dtype=torch.int)

        # --- Warmup 阶段 ---
        for i in range(num_warmup):
            input_tensor = p2p_communicator.recv_forward(
                recv_shapes, p2p_communicator.is_pp_first_stage
            )
            input_tensor = restore_tensor_from_hetero_comm(input_tensor)

            output_tensor, num_tokens = _forward_step(
                forward_step_func, data_iterator, model, input_tensor,
                forward_data_store, self.config,
                cp_group_size=cp_size,
                is_last_stage=p2p_communicator.is_pp_last_stage,
                current_microbatch=i,
            )
            total_num_tokens += num_tokens

            output_to_send = prepare_tensor_for_hetero_comm(output_tensor)
            p2p_communicator.send_forward(output_to_send, p2p_communicator.is_pp_last_stage)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                _deallocate(output_tensor, self.config)

            logger.debug("Warmup step %d/%d done", i + 1, num_warmup)

        # 稳态首次 recv
        if num_remaining > 0:
            input_tensor = p2p_communicator.recv_forward(
                recv_shapes, p2p_communicator.is_pp_first_stage
            )
            input_tensor = restore_tensor_from_hetero_comm(input_tensor)

        # --- 1F1B 稳态 ---
        for i in range(num_remaining):
            last_iteration = i == num_remaining - 1

            output_tensor, num_tokens = _forward_step(
                forward_step_func, data_iterator, model, input_tensor,
                forward_data_store, self.config,
                cp_group_size=cp_size,
                is_last_stage=p2p_communicator.is_pp_last_stage,
                current_microbatch=i + num_warmup,
            )
            total_num_tokens += num_tokens

            if forward_only:
                output_to_send = prepare_tensor_for_hetero_comm(output_tensor)
                p2p_communicator.send_forward(output_to_send, p2p_communicator.is_pp_last_stage)
                if not last_iteration:
                    input_tensor = p2p_communicator.recv_forward(
                        recv_shapes, p2p_communicator.is_pp_first_stage
                    )
                    input_tensor = restore_tensor_from_hetero_comm(input_tensor)
            else:
                output_to_send = prepare_tensor_for_hetero_comm(output_tensor)
                output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                    output_to_send, recv_shapes, p2p_communicator.is_pp_last_stage
                )
                output_tensor_grad = restore_tensor_from_hetero_comm(output_tensor_grad)

                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                _deallocate(output_tensor, self.config)

                bwd_input = input_tensors.pop(0)
                bwd_output = output_tensors.pop(0)

                if num_warmup == 0 and last_iteration:
                    _enable_grad_sync(self.config, p2p_communicator)

                input_tensor_grad = backward_func(
                    bwd_input, bwd_output, output_tensor_grad, self.config
                )

                if last_iteration:
                    input_tensor = None
                    bwd_to_send = prepare_tensor_for_hetero_comm(input_tensor_grad)
                    p2p_communicator.send_backward(
                        bwd_to_send, p2p_communicator.is_pp_first_stage
                    )
                else:
                    bwd_to_send = prepare_tensor_for_hetero_comm(input_tensor_grad)
                    input_tensor = p2p_communicator.send_backward_recv_forward(
                        bwd_to_send, recv_shapes, p2p_communicator.is_pp_first_stage
                    )
                    input_tensor = restore_tensor_from_hetero_comm(input_tensor)

            logger.debug("1F1B step %d/%d done", i + 1, num_remaining)

        # --- Cooldown 阶段 ---
        if not forward_only:
            for i in range(num_warmup):
                if i == num_warmup - 1:
                    _enable_grad_sync(self.config, p2p_communicator)

                bwd_input = input_tensors.pop(0)
                bwd_output = output_tensors.pop(0)

                output_tensor_grad = p2p_communicator.recv_backward(
                    recv_shapes, p2p_communicator.is_pp_last_stage
                )
                output_tensor_grad = restore_tensor_from_hetero_comm(output_tensor_grad)

                input_tensor_grad = backward_func(
                    bwd_input, bwd_output, output_tensor_grad, self.config
                )

                bwd_to_send = prepare_tensor_for_hetero_comm(input_tensor_grad)
                p2p_communicator.send_backward(
                    bwd_to_send, p2p_communicator.is_pp_first_stage
                )

                logger.debug("Cooldown step %d/%d done", i + 1, num_warmup)

        # finalize grads
        if getattr(self.config, "finalize_model_grads_func", None) and not forward_only:
            self.config.finalize_model_grads_func(
                model, total_num_tokens
            )

        return forward_data_store

    def teardown(self) -> None:
        """释放所有资源：SLC 缓存 + PG 缓存。

        应在训练结束或模型切换时调用，与上游 HyperCommGrid.destroy() 和
        BridgeCommunicator.destroy_broadcast_pgs() 对应。
        """
        self.slc.clear()
        BroadcastPGCache.destroy_all()
        logger.info("HeteroMultimodulePipeline torn down")


# ---------------------------------------------------------------------------
# 9. 内部辅助函数
# ---------------------------------------------------------------------------

def _forward_step(
    forward_step_func: Callable,
    data_iterator,
    model,
    input_tensor,
    forward_data_store: List,
    config,
    cp_group_size: Optional[int],
    is_last_stage: bool,
    current_microbatch: int,
) -> Tuple[any, torch.Tensor]:
    """封装单步前向传播，处理 loss 计算和数据收集。

    Args:
        forward_step_func: 用户提供的前向函数
        data_iterator: 数据迭代器
        model: 模型
        input_tensor: 来自上游 stage 的输入张量（可为 None/dict/tensor）
        forward_data_store: 收集 loss_reduced 的列表
        config: 配置
        cp_group_size: CP size（None 时不做 loss 缩放）
        is_last_stage: 是否最后阶段
        current_microbatch: 当前 micro-batch 编号

    Returns:
        (output_tensor, num_tokens)
    """
    if isinstance(model, list):
        model = model[0]

    if hasattr(model, "set_input_tensor") and input_tensor is not None:
        tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        model.set_input_tensor(tensors)

    output_tensor, loss_func = forward_step_func(data_iterator, model)

    num_tokens = torch.tensor(0, dtype=torch.int)

    if is_last_stage and loss_func is not None:
        output_tensor_dict = output_tensor if isinstance(output_tensor, dict) else output_tensor
        loss, loss_reduced = loss_func(output_tensor_dict)

        if isinstance(loss_reduced, dict):
            forward_data_store.append(loss_reduced)

        # CP loss scaling
        if cp_group_size is not None and cp_group_size > 1:
            if not getattr(config, "calculate_per_token_loss", False):
                num_mb = getattr(config, "_num_microbatches", 1)
                loss = loss * cp_group_size / num_mb

        output_tensor = loss

    return output_tensor, num_tokens


def _backward_step_single_module(
    input_tensor,
    output_tensor,
    output_tensor_grad,
    config,
) -> any:
    """单模块反向传播（非多模块场景的简化版）。

    对应上游 backward_step，去掉了已废弃的 model_type 参数。
    """
    if input_tensor is not None:
        t = input_tensor[0] if isinstance(input_tensor, list) else input_tensor
        if t is not None:
            t.retain_grad()

    if output_tensor_grad is None and config.grad_scale_func is not None:
        if isinstance(output_tensor, list):
            output_tensor[0] = config.grad_scale_func(output_tensor[0])
        else:
            output_tensor = config.grad_scale_func(output_tensor)

    out = output_tensor[0] if isinstance(output_tensor, list) else output_tensor
    out_grad = output_tensor_grad[0] if isinstance(output_tensor_grad, list) else output_tensor_grad

    if out is not None and out.requires_grad:
        if getattr(config, "deallocate_pipeline_outputs", False):
            _custom_backward(out, out_grad)
        else:
            torch.autograd.backward(out, grad_tensors=out_grad)

    if input_tensor is None:
        return None
    t = input_tensor[0] if isinstance(input_tensor, list) else input_tensor
    return t.grad if t is not None else None


def _deallocate(output_tensor, config) -> None:
    """释放已发送激活的数据（保留 grad_fn）。

    上游 deallocate_output_tensor 的 DES-LOC 版本，
    扩展支持 dict（多模块）和 list（VPP）格式。
    """
    if not getattr(config, "deallocate_pipeline_outputs", False):
        return
    if output_tensor is None:
        return
    if isinstance(output_tensor, dict):
        for v in output_tensor.values():
            _deallocate(v, config)
        return
    if isinstance(output_tensor, list):
        for item in output_tensor:
            _deallocate(item, config)
        return
    if isinstance(output_tensor, torch.Tensor):
        assert output_tensor._base is None, "counter-productive to free a view"
        output_tensor.data = torch.empty(
            (1,), device=output_tensor.device, dtype=output_tensor.dtype
        )


def _enable_grad_sync(config, p2p_communicator) -> None:
    """在最后一个 micro-batch 时启用梯度同步。"""
    if getattr(config, "grad_sync_func", None) is None or p2p_communicator.is_pp_first_stage:
        if hasattr(config, "_no_sync_context") and config._no_sync_context is not None:
            config._no_sync_context.__exit__(None, None, None)
            config._no_sync_context = None
            logger.debug("Grad sync re-enabled at pp_first_stage")


# ---------------------------------------------------------------------------
# 10. Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # --- Test 1: HeteroDeviceProfile ---
    p = HeteroDeviceProfile(
        device_index=0, sm_major=8, sm_minor=6, vram_gb=48.0, pcie_bw_gbps=32.0, role="encoder"
    )
    assert p.sm_version == 86
    assert p.is_ampere
    assert not p.is_hopper
    logger.info("Test 1 passed: HeteroDeviceProfile A6000 SM86")

    p2 = HeteroDeviceProfile(
        device_index=2, sm_major=9, sm_minor=0, vram_gb=96.0, pcie_bw_gbps=64.0, role="llm"
    )
    assert p2.sm_version == 90
    assert p2.is_hopper
    logger.info("Test 2 passed: HeteroDeviceProfile H100 SM90")

    # --- Test 2: SLC store/retrieve ---
    if torch.cuda.is_available():
        slc = SharedLocalityCache(max_entries=8)
        t = torch.randn(4, 8, 16, device="cuda")
        slc.store("encoder", microbatch_id=0, tensor=t, direction="fwd")
        torch.cuda.synchronize()
        t_back = slc.retrieve("encoder", microbatch_id=0, target_device=torch.device("cuda"), direction="fwd")
        assert t_back is not None
        assert t_back.shape == t.shape
        assert torch.allclose(t_back, t, atol=1e-5)
        slc.clear()
        assert slc.size == 0
        logger.info("Test 3 passed: SLC GPU roundtrip (store → retrieve)")
    else:
        logger.info("Test 3 skipped: no CUDA device")

    # --- Test 3: prepare/restore tensor ---
    x2d = torch.randn(4, 8)
    x3d = prepare_tensor_for_hetero_comm(x2d)
    assert x3d.shape == (4, 8, 1)
    x2d_restored = restore_tensor_from_hetero_comm(x3d)
    assert x2d_restored.shape == (4, 8)
    assert torch.allclose(x2d, x2d_restored)
    logger.info("Test 4 passed: prepare/restore 2D↔3D tensor")

    # --- Test 4: DESLOCBridgeConfig auto resolution ---
    cfg = DESLOCBridgeConfig(src_module="encoder", dst_module="llm", transport="auto", slc_threshold_mb=64.0)
    transport = cfg.resolve_transport(tensor_mb=128.0, src_profile=p, dst_profile=p2)
    assert transport == "slc", f"Expected slc (cross-SM + large tensor), got {transport}"
    transport_small = cfg.resolve_transport(tensor_mb=10.0, src_profile=p, dst_profile=p2)
    assert transport_small == "slc", "Cross-SM should always be slc"
    cfg_same_sm = DESLOCBridgeConfig(src_module="enc1", dst_module="enc2", transport="auto")
    transport_nccl = cfg_same_sm.resolve_transport(tensor_mb=10.0, src_profile=p, dst_profile=p)
    assert transport_nccl == "nccl", f"Same-SM small tensor should be nccl, got {transport_nccl}"
    logger.info("Test 5 passed: DESLOCBridgeConfig transport resolution")

    # --- Test 5: HeteroModuleProcessGroups ---
    mock_pg = object()  # placeholder, no real dist needed for structure test
    hmpg = HeteroModuleProcessGroups(
        module_pgs={"encoder": {"tp": mock_pg, "cp": mock_pg}, "llm": {"tp": mock_pg, "cp": mock_pg}},
        language_model_module_name="llm",
        device_profiles={0: p, 1: p, 2: p2},
    )
    assert hmpg.has_language_model()
    assert len(hmpg) == 2
    assert list(hmpg.keys()) == ["encoder", "llm"]
    logger.info("Test 6 passed: HeteroModuleProcessGroups structure")

    logger.info("All smoke tests passed.")
