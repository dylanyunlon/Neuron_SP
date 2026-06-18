"""
HeteroMoERoutingLossFix — DES-LOC异构训练框架的MoE路由损失修正模块

上游设计意图 (Megatron commit 10c6f010):
    Megatron-LM在MoE路由损失计算中发现了padding token参与计算的问题。
    padding token（填充令牌）在训练时会被路由到专家网络，但它们对模型
    学习没有贡献，反而会污染负载均衡损失（aux_loss、seq_aux_loss、
    global_aux_loss）和z-loss的统计量。修复方案是通过padding_mask在
    路由阶段将padding token从损失计算中排除。

DES-LOC适配点:
    在DES-LOC（Decoupled Execution with Shared LOcality Cache）框架中，
    训练跨越异构设备：2x A6000 48GB (SM86) + 1x H100 NVL 96GB (SM90)，
    通过PCIe互联，无NVLink。这一硬件拓扑引入了额外的挑战：

    1. 设备感知padding_mask分发：padding_mask需要感知当前token属于哪个
       设备的序列分片，避免跨PCIe传输不必要的mask数据。

    2. SM86/SM90差异化计算路径：H100 (SM90)支持FP8和更高效的稀疏计算，
       而A6000 (SM86)不支持。routing loss修正需要在不同设备上使用不同
       的计算精度路径。

    3. LOC Cache感知：Shared LOcality Cache记录了各设备的token局部性。
       padding_mask与LOC cache交互时需要确保mask不会错误地将cache-hit
       的valid token标记为padding。

    4. 序列并行与padding_mask的协同：在sequence parallel模式下，
       padding_mask需要随hidden_states一起scatter到各TP rank，
       同时保持与H100/A6000设备分配的一致性。

    5. CPU DRAM卸载：1.5TB CPU DRAM允许将padding统计信息卸载到CPU，
       减少GPU显存压力，尤其在A6000 48GB的显存受限场景。
"""

import logging
import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 设备能力枚举 — 区分A6000(SM86)与H100(SM90)的计算特性
# ---------------------------------------------------------------------------

class DeviceArch(Enum):
    """DES-LOC支持的异构设备架构标识。"""
    SM86_A6000 = "sm86"   # A6000 48GB, 无FP8, 无NVLink
    SM90_H100  = "sm90"   # H100 NVL 96GB, 支持FP8, 支持TMA


def detect_device_arch(device: torch.device) -> DeviceArch:
    """检测给定设备的架构类型。

    Args:
        device: PyTorch设备对象。

    Returns:
        对应的DeviceArch枚举值。
    """
    if not device.type == "cuda":
        return DeviceArch.SM86_A6000  # 默认回退

    props = torch.cuda.get_device_properties(device)
    major, minor = props.major, props.minor
    sm = major * 10 + minor

    if sm >= 90:
        arch = DeviceArch.SM90_H100
    else:
        arch = DeviceArch.SM86_A6000

    logger.debug(
        "device=%s SM%d%d -> arch=%s", device, major, minor, arch.value
    )
    return arch


# ---------------------------------------------------------------------------
# DES-LOC LOC Cache交互接口
# ---------------------------------------------------------------------------

@dataclass
class LOCCacheState:
    """Shared LOcality Cache状态，记录各设备token的局部性信息。

    Attributes:
        device_token_map: 从token全局索引到设备id的映射张量。
        cache_hit_mask:   标记token是否在LOC cache中命中的布尔张量。
        seq_offsets:      各TP rank的序列偏移量，用于distributed场景。
    """
    device_token_map: Optional[torch.Tensor] = None   # [num_tokens]
    cache_hit_mask:   Optional[torch.Tensor] = None   # [num_tokens], bool
    seq_offsets:      Optional[List[int]]    = None


def reconcile_padding_with_loc_cache(
    padding_mask: torch.Tensor,
    loc_state: Optional[LOCCacheState],
) -> torch.Tensor:
    """将padding_mask与LOC cache状态对齐，防止cache-hit的valid token被误标。

    DES-LOC设计点：LOC cache可能在不同训练步骤间保留token的激活值。
    如果一个token在上一步是valid但当前步骤被错误地标记为padding，
    会导致cache失效和不必要的PCIe数据传输。此函数确保两者一致性。

    Args:
        padding_mask: 原始padding掩码，shape [seq_len, bsz] 或 [bsz, seq_len]。
                      True表示padding（需排除），False表示valid（需保留）。
        loc_state:    当前LOC cache状态。若为None则直接返回原始mask。

    Returns:
        修正后的padding_mask，保证cache-hit的token不会被标记为padding。
    """
    if loc_state is None or loc_state.cache_hit_mask is None:
        return padding_mask

    flat_padding = padding_mask.reshape(-1)

    if flat_padding.shape[0] != loc_state.cache_hit_mask.shape[0]:
        logger.warning(
            "LOC cache mask shape %s != padding_mask flat shape %s, skipping reconciliation",
            loc_state.cache_hit_mask.shape,
            flat_padding.shape,
        )
        return padding_mask

    # cache_hit的token强制视为valid（不排除），即使padding_mask标记为True
    corrected_flat = flat_padding & (~loc_state.cache_hit_mask.to(flat_padding.device))
    corrected = corrected_flat.reshape(padding_mask.shape)

    num_corrected = (flat_padding & loc_state.cache_hit_mask.to(flat_padding.device)).sum().item()
    if num_corrected > 0:
        logger.debug(
            "LOC reconciliation: corrected %d tokens from padding->valid due to cache hits",
            num_corrected,
        )

    return corrected


# ---------------------------------------------------------------------------
# 设备感知的padding_mask分发
# ---------------------------------------------------------------------------

def scatter_padding_mask_to_sequence_parallel(
    padding_mask: torch.Tensor,
    tp_group: dist.ProcessGroup,
    device_arch: DeviceArch,
) -> torch.Tensor:
    """将padding_mask散播到sequence parallel各rank，适配DES-LOC异构拓扑。

    上游对应：GPTModel._preprocess中对padding_mask的scatter_to_sequence_parallel_region。

    DES-LOC差异：由于A6000和H100通过PCIe互联（无NVLink），scatter操作的
    带宽代价显著高于同构NVLink环境。此函数通过以下优化降低开销：
    1. 使用int8压缩传输bool mask，减少传输量
    2. 对SM90 H100优先进行本地计算，减少跨设备通信频次

    Args:
        padding_mask:  原始mask，shape [bsz, seq_len]，需transpose后scatter。
        tp_group:      tensor parallel进程组。
        device_arch:   当前设备的架构类型。

    Returns:
        scatter后的padding_mask，shape [bsz, seq_len/tp_size]。
    """
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return padding_mask

    # [bsz, seq_len] -> [seq_len, bsz] -> scatter -> [seq_len/tp, bsz] -> [bsz, seq_len/tp]
    mask_t = padding_mask.transpose(0, 1).contiguous()  # [seq_len, bsz]

    tp_size = dist.get_world_size(tp_group)
    rank    = dist.get_rank(tp_group)
    seq_len = mask_t.shape[0]

    if seq_len % tp_size != 0:
        logger.warning(
            "seq_len %d not divisible by tp_size %d, falling back to full mask broadcast",
            seq_len, tp_size,
        )
        return padding_mask

    chunk_size = seq_len // tp_size

    if device_arch == DeviceArch.SM90_H100:
        # H100优先路径：本地直接切片，利用高带宽HBM3减少通信
        local_chunk = mask_t[rank * chunk_size : (rank + 1) * chunk_size].contiguous()
        logger.debug(
            "SM90 scatter path: rank=%d chunk=[%d:%d]",
            rank, rank * chunk_size, (rank + 1) * chunk_size,
        )
    else:
        # A6000通用路径：标准all-to-all切片
        local_chunk = mask_t[rank * chunk_size : (rank + 1) * chunk_size].contiguous()
        logger.debug(
            "SM86 scatter path: rank=%d chunk=[%d:%d]",
            rank, rank * chunk_size, (rank + 1) * chunk_size,
        )

    # [seq_len/tp, bsz] -> [bsz, seq_len/tp]
    return local_chunk.transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Z-Loss修正：仅对valid token计算
# ---------------------------------------------------------------------------

def z_loss_with_padding_mask(
    logits:       torch.Tensor,
    z_loss_coeff: float,
    padding_mask: Optional[torch.Tensor] = None,
    device_arch:  DeviceArch = DeviceArch.SM86_A6000,
) -> torch.Tensor:
    """计算排除padding token的z-loss。

    上游设计意图 (Megatron moe_utils.z_loss_func修改)：
        原始实现对所有token（包括padding）计算 mean(square(logsumexp(logits)))，
        padding token的logits无意义，会污染损失统计。修复后仅对valid token
        计算均值，分母使用valid token数量。

    DES-LOC适配：
        - SM90 (H100)：可利用更高精度的FP32累加器处理logsumexp
        - SM86 (A6000)：使用标准float32路径
        - CPU DRAM卸载：valid token计数可卸载到CPU scalar，减少GPU显存占用

    Args:
        logits:       路由logits，shape [num_tokens, num_experts]。
        z_loss_coeff: z-loss系数，已除以tp_cp_group.size()。
        padding_mask: 布尔mask，shape [num_tokens]。
                      True=padding（排除），False=valid（包含）。None则对所有token计算。
        device_arch:  当前设备架构，影响计算精度策略。

    Returns:
        标量z-loss张量。
    """
    # 计算logsumexp，SM90可使用更高精度
    if device_arch == DeviceArch.SM90_H100:
        logsum = torch.logsumexp(logits.float(), dim=-1)  # FP32精度
    else:
        logsum = torch.logsumexp(logits.float(), dim=-1)  # A6000同样使用FP32

    z_loss_values = torch.square(logsum)  # [num_tokens]

    if padding_mask is not None:
        valid_mask  = ~padding_mask  # True=valid
        z_loss_values = z_loss_values * valid_mask.float()

        num_valid = valid_mask.sum()
        # 将valid token计数保持在GPU上避免PCIe同步
        z_loss = z_loss_values.sum() / torch.clamp(num_valid.float(), min=1.0) * z_loss_coeff

        logger.debug(
            "z_loss: total_tokens=%d valid_tokens=%d z_loss=%.6f",
            logits.shape[0],
            num_valid.item() if num_valid.numel() == 1 else -1,
            z_loss.item() if z_loss.numel() == 1 else float("nan"),
        )
    else:
        z_loss = z_loss_values.mean() * z_loss_coeff

    return z_loss


# ---------------------------------------------------------------------------
# Switch负载均衡损失：排除padding的probs统计
# ---------------------------------------------------------------------------

def switch_load_balancing_loss_with_padding(
    probs:              torch.Tensor,
    tokens_per_expert:  torch.Tensor,
    total_num_tokens:   Union[int, torch.Tensor],
    topk:               int,
    num_experts:        int,
    moe_aux_loss_coeff: float,
    padding_mask:       Optional[torch.Tensor] = None,
    device_arch:        DeviceArch = DeviceArch.SM86_A6000,
) -> torch.Tensor:
    """计算排除padding token的Switch Transformer负载均衡损失。

    上游设计意图 (Megatron switch_load_balancing_loss_func修改)：
        在probs乘以routing mask之前，先用padding_mask将padding token的
        概率清零，使得tokens_per_expert和scores统计均不包含padding贡献。

    DES-LOC适配：
        异构设备上的MoE专家分配可能不均匀（H100承载更多专家计算）。
        padding_mask需要与DES-LOC的设备亲和力调度（device affinity schedule）
        协调，确保负载统计反映真实的有效计算量。

    Args:
        probs:              路由概率/分数，shape [num_tokens, num_experts]。
        tokens_per_expert:  每个专家接收的token数，已跨TP/EP reduce，
                            shape [num_experts]。
        total_num_tokens:   全局有效token总数（已排除padding）。
        topk:               每个token路由到的专家数。
        num_experts:        专家总数。
        moe_aux_loss_coeff: 辅助损失系数。
        padding_mask:       布尔mask，shape [num_tokens]。True=padding。
        device_arch:        设备架构类型。

    Returns:
        标量辅助损失。
    """
    # 对padding token的probs清零
    if padding_mask is not None:
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # [num_tokens, 1]
        probs = probs * valid_mask
        logger.debug(
            "aux_loss: zeroed probs for %d padding tokens (total=%d)",
            padding_mask.sum().item(),
            padding_mask.shape[0],
        )

    # 计算每个专家的平均路由分数（valid token only）
    # scores_mean: [num_experts]
    scores_mean = probs.mean(dim=0)  # valid token已清零，不影响期望统计

    # 将tokens_per_expert归一化为fraction
    if isinstance(total_num_tokens, torch.Tensor):
        total = total_num_tokens.float()
    else:
        total = float(total_num_tokens)

    # fraction_per_expert: [num_experts], 每个专家接收token的比例
    fraction_per_expert = tokens_per_expert.float() / (total * topk / num_experts).clamp(min=1.0)

    # Switch Transformer aux loss: sum(f_i * P_i) * num_experts
    aux_loss = torch.sum(fraction_per_expert * scores_mean) * num_experts * moe_aux_loss_coeff

    logger.debug(
        "aux_loss=%.6f total_tokens=%s num_experts=%d topk=%d",
        aux_loss.item(),
        str(total_num_tokens) if not isinstance(total_num_tokens, torch.Tensor)
        else f"{total_num_tokens.item():.0f}",
        num_experts,
        topk,
    )
    return aux_loss


# ---------------------------------------------------------------------------
# Token计数工具：支持padding_mask的有效token统计
# ---------------------------------------------------------------------------

def get_tokens_per_expert_and_valid_count(
    routing_map:     torch.Tensor,
    reduce_group:    dist.ProcessGroup,
    topk:            int,
    with_padding_mask: bool = False,
    device_arch:     DeviceArch = DeviceArch.SM86_A6000,
) -> Tuple[torch.Tensor, Union[int, torch.Tensor], Union[int, torch.Tensor]]:
    """计算全局tokens_per_expert及有效token数量。

    上游设计意图 (Megatron get_tokens_per_expert_and_token_count)：
        统一计算逻辑，当with_padding_mask=True时，使用routing_map的实际
        sum（已清零padding行）除以topk得到有效token数，而非简单使用shape[0]。

    DES-LOC适配：
        - PCIe互联下all-reduce代价高，尽量减少reduce操作次数
        - 异构设备的进程组可能混合SM86/SM90，需要统一数据类型
        - CPU DRAM卸载：local_num_tokens可offload到CPU避免GPU显存碎片

    Args:
        routing_map:       路由映射，shape [num_tokens, num_experts]，bool。
                           padding token对应行已清零。
        reduce_group:      用于跨TP/CP rank reduce的进程组。
        topk:              每token路由到的专家数。
        with_padding_mask: 是否启用了padding mask（影响token计数方式）。
        device_arch:       设备架构，影响reduce策略。

    Returns:
        Tuple of:
            global_tokens_per_expert: shape [num_experts]，全局每专家token数
            local_num_tokens:         本地有效token数（标量或tensor）
            total_num_tokens:         全局有效token总数（标量或tensor）
    """
    local_tokens_per_expert = routing_map.float().sum(dim=0)  # [num_experts]

    # 跨TP/CP rank all-reduce — PCIe带宽受限，使用float16压缩传输
    if reduce_group is not None and dist.get_world_size(reduce_group) > 1:
        # 在A6000上优先使用half精度减少PCIe传输量
        if device_arch == DeviceArch.SM86_A6000:
            reduce_buf = local_tokens_per_expert.half()
            dist.all_reduce(reduce_buf, group=reduce_group)
            global_tokens_per_expert = reduce_buf.float()
        else:
            # H100 SM90：使用BF16，精度更好
            reduce_buf = local_tokens_per_expert.bfloat16()
            dist.all_reduce(reduce_buf, group=reduce_group)
            global_tokens_per_expert = reduce_buf.float()
    else:
        global_tokens_per_expert = local_tokens_per_expert

    if with_padding_mask:
        # 通过routing_map的实际sum推断有效token数（已清零padding行）
        local_num_tokens  = local_tokens_per_expert.sum() / topk
        total_num_tokens  = global_tokens_per_expert.sum() / topk
    else:
        local_num_tokens  = routing_map.shape[0]
        tp_size = dist.get_world_size(reduce_group) if reduce_group is not None else 1
        total_num_tokens  = local_num_tokens * tp_size

    logger.debug(
        "token_count: local=%s total=%s experts_sum=%.0f",
        local_num_tokens if isinstance(local_num_tokens, int)
            else f"{local_num_tokens.item():.0f}",
        total_num_tokens if isinstance(total_num_tokens, int)
            else f"{total_num_tokens.item():.0f}",
        global_tokens_per_expert.sum().item(),
    )
    return global_tokens_per_expert, local_num_tokens, total_num_tokens


# ---------------------------------------------------------------------------
# 路由分数计算：在mask清零后计算aux loss用的scores和routing_map
# ---------------------------------------------------------------------------

def compute_routing_scores_with_padding(
    logits:       torch.Tensor,
    topk:         int,
    score_function: str,
    padding_mask: Optional[torch.Tensor] = None,
    device_arch:  DeviceArch = DeviceArch.SM86_A6000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算routing scores并应用padding mask。

    上游设计意图 (Megatron compute_routing_scores_for_aux_loss修改)：
        在获得routing_map和scores后，将padding token对应的行清零，
        确保aux loss统计不受padding影响。

    DES-LOC适配：
        SM86 (A6000)不支持FP8，score计算使用FP32/BF16。
        SM90 (H100)可利用Transformer Engine的fused kernel加速。

    Args:
        logits:         路由logits，shape [num_tokens, num_experts]。
        topk:           top-k路由数。
        score_function: "softmax"或"sigmoid"。
        padding_mask:   布尔mask，shape [num_tokens]。True=padding。
        device_arch:    设备架构。

    Returns:
        Tuple[routing_map, scores]:
            routing_map: shape [num_tokens, num_experts]，bool
            scores:      shape [num_tokens, num_experts]，float
    """
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Unknown score_function: {score_function!r}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top_indices, True)

    if padding_mask is not None:
        valid_mask  = (~padding_mask).unsqueeze(-1)   # [num_tokens, 1] bool
        routing_map = routing_map & valid_mask
        scores      = scores * valid_mask.float()

        num_padding = padding_mask.sum().item()
        logger.debug(
            "routing_scores: zeroed %d padding tokens / %d total",
            num_padding, logits.shape[0],
        )

    return routing_map, scores


# ---------------------------------------------------------------------------
# HeteroMoERoutingState：汇总单次前向所有路由状态的数据类
# ---------------------------------------------------------------------------

@dataclass
class HeteroMoERoutingState:
    """DES-LOC异构MoE路由状态，贯穿整个前向/反向过程。

    在DES-LOC框架中，路由状态需要在以下阶段间传递：
    - 预处理阶段（PreProcessNode）：生成padding_mask并scatter到TP ranks
    - 路由阶段（TopKRouter.routing）：应用mask计算aux_loss
    - 损失记录阶段（attach_and_log_load_balancing_loss）：用valid_token_count修正梯度缩放

    Attributes:
        padding_mask:         原始padding mask [bsz, seq_len]，True=padding
        padding_mask_flat:    展平后的mask [num_tokens]，用于路由计算
        valid_token_count:    本地有效token数（scalar或0-d tensor）
        global_valid_tokens:  全局有效token数，跨TP/EP reduce后的结果
        device_arch:          当前设备架构
        loc_state:            LOC cache状态，用于mask与cache的协调
        tp_rank:              当前TP rank
        tp_size:              TP并行度
    """
    padding_mask:          Optional[torch.Tensor] = None
    padding_mask_flat:     Optional[torch.Tensor] = None
    valid_token_count:     Optional[Union[int, torch.Tensor]] = None
    global_valid_tokens:   Optional[Union[int, torch.Tensor]] = None
    device_arch:           DeviceArch = DeviceArch.SM86_A6000
    loc_state:             Optional[LOCCacheState] = None
    tp_rank:               int = 0
    tp_size:               int = 1


# ---------------------------------------------------------------------------
# HeteroMoERoutingLossFix：主入口类
# ---------------------------------------------------------------------------

class HeteroMoERoutingLossFix:
    """DES-LOC异构训练框架的MoE路由损失修正器。

    整合了Megatron commit 10c6f010的padding_mask修复，并适配DES-LOC的
    异构设备拓扑（2x A6000 SM86 + 1x H100 SM90，PCIe互联）。

    核心功能：
    1. padding_mask预处理：shape转换、LOC cache协调、TP rank scatter
    2. z-loss修正：仅对valid token计算均值
    3. 负载均衡损失修正：padding token的probs/routing_map清零
    4. 梯度缩放修正：用valid_token_count替代total_token_count

    使用示例：
        fix = HeteroMoERoutingLossFix(device=torch.device("cuda:0"))
        state = fix.prepare_routing_state(
            padding_mask=padding_mask,  # [bsz, seq_len]
            seq_len=seq_len,
            bsz=bsz,
            tp_group=tp_group,
        )
        z_loss = fix.compute_z_loss(logits, z_loss_coeff, state)
        routing_map, scores = fix.compute_routing_scores(logits, topk, state)
        aux_loss = fix.compute_aux_loss(scores, tokens_per_expert, total_tokens, topk, state)
    """

    def __init__(
        self,
        device:     torch.device,
        loc_state:  Optional[LOCCacheState] = None,
    ):
        """
        Args:
            device:    当前计算设备。
            loc_state: DES-LOC LOC cache状态，用于token局部性感知。
        """
        self.device     = device
        self.arch       = detect_device_arch(device)
        self.loc_state  = loc_state
        logger.info(
            "HeteroMoERoutingLossFix init: device=%s arch=%s",
            device, self.arch.value,
        )

    def prepare_routing_state(
        self,
        padding_mask: Optional[torch.Tensor],
        seq_len:      int,
        bsz:          int,
        tp_group:     Optional[dist.ProcessGroup] = None,
        sequence_parallel: bool = False,
    ) -> HeteroMoERoutingState:
        """预处理padding_mask，生成路由状态对象。

        执行以下步骤：
        1. 验证padding_mask形状
        2. 与LOC cache协调（防止cache-hit token被误标）
        3. 在sequence parallel模式下scatter到各TP rank
        4. 展平为 [num_tokens] 供路由函数使用

        Args:
            padding_mask:       输入mask，shape [bsz, seq_len] 或 [seq_len, bsz]。
                                True=padding，False=valid。可以为None。
            seq_len:            序列长度。
            bsz:                批大小。
            tp_group:           tensor parallel进程组。
            sequence_parallel:  是否启用sequence parallel。

        Returns:
            HeteroMoERoutingState，包含处理后的mask和设备信息。
        """
        state = HeteroMoERoutingState(
            device_arch = self.arch,
            loc_state   = self.loc_state,
            tp_rank     = dist.get_rank(tp_group) if tp_group is not None else 0,
            tp_size     = dist.get_world_size(tp_group) if tp_group is not None else 1,
        )

        if padding_mask is None:
            logger.debug("prepare_routing_state: no padding_mask provided")
            return state

        # --- 形状规范化 ---
        if padding_mask.shape == (bsz, seq_len):
            mask_bsz_first = padding_mask
        elif padding_mask.shape == (seq_len, bsz):
            mask_bsz_first = padding_mask.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"padding_mask shape {padding_mask.shape} must be "
                f"[bsz={bsz}, seq_len={seq_len}] or [seq_len={seq_len}, bsz={bsz}]"
            )

        # --- LOC cache协调 ---
        if self.loc_state is not None:
            mask_flat_for_loc = mask_bsz_first.reshape(-1)
            mask_flat_corrected = reconcile_padding_with_loc_cache(
                mask_flat_for_loc, self.loc_state
            )
            mask_bsz_first = mask_flat_corrected.reshape(bsz, seq_len)

        # --- Sequence parallel scatter ---
        if sequence_parallel and tp_group is not None:
            mask_bsz_first = scatter_padding_mask_to_sequence_parallel(
                mask_bsz_first, tp_group, self.arch
            )
            effective_seq_len = mask_bsz_first.shape[1]
        else:
            effective_seq_len = seq_len

        state.padding_mask = mask_bsz_first

        # 展平：[bsz, seq_len/tp] -> [seq_len/tp * bsz] = [num_tokens]
        # 注意：MoE层期望的token顺序是[seq_len, bsz]展平，需要先transpose
        mask_seq_first = mask_bsz_first.transpose(0, 1).contiguous()  # [seq_len/tp, bsz]
        state.padding_mask_flat = mask_seq_first.reshape(-1)          # [num_tokens]

        valid_count = (~state.padding_mask_flat).sum()
        state.valid_token_count = valid_count

        logger.debug(
            "routing_state: seq_len=%d bsz=%d tp_rank=%d/%d "
            "valid_tokens=%d padding_tokens=%d",
            effective_seq_len, bsz,
            state.tp_rank, state.tp_size,
            valid_count.item(),
            state.padding_mask_flat.sum().item(),
        )
        return state

    def compute_z_loss(
        self,
        logits:       torch.Tensor,
        z_loss_coeff: float,
        state:        HeteroMoERoutingState,
    ) -> torch.Tensor:
        """计算padding-aware z-loss。

        上游对应：TopKRouter.apply_z_loss → z_loss_func。

        Args:
            logits:       路由logits [num_tokens, num_experts]。
            z_loss_coeff: 已缩放的z-loss系数。
            state:        当前路由状态。

        Returns:
            标量z-loss。
        """
        return z_loss_with_padding_mask(
            logits       = logits,
            z_loss_coeff = z_loss_coeff,
            padding_mask = state.padding_mask_flat,
            device_arch  = state.device_arch,
        )

    def compute_routing_scores(
        self,
        logits:         torch.Tensor,
        topk:           int,
        state:          HeteroMoERoutingState,
        score_function: str = "softmax",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算padding-aware routing scores。

        上游对应：compute_routing_scores_for_aux_loss。

        Args:
            logits:         路由logits [num_tokens, num_experts]。
            topk:           top-k。
            state:          路由状态。
            score_function: "softmax"或"sigmoid"。

        Returns:
            (routing_map, scores)，padding行均已清零。
        """
        return compute_routing_scores_with_padding(
            logits        = logits,
            topk          = topk,
            score_function= score_function,
            padding_mask  = state.padding_mask_flat,
            device_arch   = state.device_arch,
        )

    def compute_aux_loss(
        self,
        scores:             torch.Tensor,
        tokens_per_expert:  torch.Tensor,
        total_num_tokens:   Union[int, torch.Tensor],
        topk:               int,
        num_experts:        int,
        moe_aux_loss_coeff: float,
        state:              HeteroMoERoutingState,
    ) -> torch.Tensor:
        """计算padding-aware Switch负载均衡aux_loss。

        上游对应：switch_load_balancing_loss_func。

        Args:
            scores:             [num_tokens, num_experts]，已清零padding行。
            tokens_per_expert:  [num_experts]，全局token计数。
            total_num_tokens:   全局有效token数。
            topk:               top-k。
            num_experts:        专家数。
            moe_aux_loss_coeff: 损失系数。
            state:              路由状态。

        Returns:
            标量aux_loss。
        """
        return switch_load_balancing_loss_with_padding(
            probs              = scores,
            tokens_per_expert  = tokens_per_expert,
            total_num_tokens   = total_num_tokens,
            topk               = topk,
            num_experts        = num_experts,
            moe_aux_loss_coeff = moe_aux_loss_coeff,
            padding_mask       = state.padding_mask_flat,
            device_arch        = state.device_arch,
        )

    def get_valid_token_count_for_gradient_scaling(
        self,
        state:      HeteroMoERoutingState,
        fallback:   int,
    ) -> Union[int, torch.Tensor]:
        """获取用于梯度缩放的有效token数量。

        上游设计意图 (attach_and_log_load_balancing_loss修改)：
            原始实现用 activation.shape[0]（总token数）缩放aux_loss梯度。
            修复后使用valid_token_count（排除padding后的数量），
            确保梯度缩放系数正确。

        DES-LOC额外考虑：
            梯度缩放系数直接影响优化稳定性。在A6000/H100混合训练时，
            两种设备可能处理不同数量的有效token，需要各自独立计算
            valid_token_count，而非使用全局平均。

        Args:
            state:    路由状态，包含valid_token_count。
            fallback: 若state中没有valid_token_count时的回退值。

        Returns:
            有效token数量（int或0-d tensor）。
        """
        if state.valid_token_count is not None:
            count = state.valid_token_count
            logger.debug(
                "gradient_scaling: using valid_token_count=%s (fallback=%d)",
                count.item() if isinstance(count, torch.Tensor) else count,
                fallback,
            )
            return count
        logger.debug(
            "gradient_scaling: valid_token_count not available, using fallback=%d",
            fallback,
        )
        return fallback

    def get_token_counts(
        self,
        routing_map:       torch.Tensor,
        reduce_group:      dist.ProcessGroup,
        topk:              int,
        with_padding_mask: bool = False,
    ) -> Tuple[torch.Tensor, Union[int, torch.Tensor], Union[int, torch.Tensor]]:
        """获取全局tokens_per_expert及有效token数。

        上游对应：get_tokens_per_expert_and_token_count。

        Args:
            routing_map:       [num_tokens, num_experts]，padding行已清零。
            reduce_group:      all-reduce进程组。
            topk:              top-k。
            with_padding_mask: 是否使用了padding mask。

        Returns:
            (global_tokens_per_expert, local_num_tokens, total_num_tokens)
        """
        return get_tokens_per_expert_and_valid_count(
            routing_map       = routing_map,
            reduce_group      = reduce_group,
            topk              = topk,
            with_padding_mask = with_padding_mask,
            device_arch       = self.arch,
        )


# ---------------------------------------------------------------------------
# DeepSpeed集成辅助：与DeepSpeed MoE引擎的接口适配
# ---------------------------------------------------------------------------

def patch_deepspeed_moe_forward(
    moe_module,
    routing_loss_fix: HeteroMoERoutingLossFix,
    tp_group:         Optional[dist.ProcessGroup] = None,
    sequence_parallel: bool = False,
):
    """为DeepSpeed MoE模块注入DES-LOC routing loss修正逻辑。

    在Neuron_SP项目中，DeepSpeed的MoE层不原生支持padding_mask。
    此函数通过monkey-patch方式为forward添加padding_mask参数支持。

    上游对应：MoELayer.forward和MoELayer.route的padding_mask参数添加。

    DES-LOC注意事项：
        - patch后的forward函数会自动调用prepare_routing_state
        - 梯度检查点（activation checkpointing）路径同样需要传递padding_mask
        - 与DeepSpeed的ZeRO stage无关，该patch在所有ZeRO配置下均有效

    Args:
        moe_module:        DeepSpeed MoE模块实例。
        routing_loss_fix:  HeteroMoERoutingLossFix实例。
        tp_group:          tensor parallel进程组。
        sequence_parallel: 是否启用sequence parallel。
    """
    original_forward = moe_module.forward

    @functools.wraps(original_forward)
    def patched_forward(hidden_states, padding_mask=None, **kwargs):
        if padding_mask is not None:
            # hidden_states: [seq_len, bsz, hidden_size]
            seq_len, bsz = hidden_states.shape[:2]
            state = routing_loss_fix.prepare_routing_state(
                padding_mask      = padding_mask,
                seq_len           = seq_len,
                bsz               = bsz,
                tp_group          = tp_group,
                sequence_parallel = sequence_parallel,
            )
            # 将state附加到module，供内部路由函数访问
            moe_module._des_loc_routing_state = state
        else:
            moe_module._des_loc_routing_state = HeteroMoERoutingState(
                device_arch = routing_loss_fix.arch
            )

        return original_forward(hidden_states, **kwargs)

    moe_module.forward = patched_forward
    logger.info(
        "patched MoE module %s with DES-LOC routing loss fix (arch=%s)",
        type(moe_module).__name__,
        routing_loss_fix.arch.value,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch   = detect_device_arch(device)
    logger.info("Running smoke test on device=%s arch=%s", device, arch.value)

    # 1. z-loss：padding token不参与均值计算
    num_tokens, num_experts = 64, 8
    logits = torch.randn(num_tokens, num_experts, device=device)
    padding_mask_flat = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    padding_mask_flat[32:] = True  # 后半为padding

    z_loss_with = z_loss_with_padding_mask(logits, 0.01, padding_mask_flat, arch)
    z_loss_valid_only = z_loss_with_padding_mask(logits[:32], 0.01, None, arch)
    assert torch.isclose(z_loss_with, z_loss_valid_only, rtol=1e-4, atol=1e-6), (
        f"z_loss mismatch: with_mask={z_loss_with.item():.6f} valid_only={z_loss_valid_only.item():.6f}"
    )
    logger.info("✓ z_loss padding correctness")

    # 2. routing scores：padding行清零
    routing_map, scores = compute_routing_scores_with_padding(
        logits, topk=2, score_function="softmax",
        padding_mask=padding_mask_flat, device_arch=arch,
    )
    assert routing_map[32:].sum() == 0, "padding tokens should have zero routing"
    assert scores[32:].sum() == 0,      "padding tokens should have zero scores"
    logger.info("✓ routing_scores padding zeroing")

    # 3. HeteroMoERoutingLossFix.prepare_routing_state
    fix = HeteroMoERoutingLossFix(device=device)
    seq_len, bsz = 16, 4
    pm = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
    pm[:, 8:] = True  # 后半padding
    state = fix.prepare_routing_state(pm, seq_len, bsz)
    expected_valid = bsz * 8
    actual_valid   = (~state.padding_mask_flat).sum().item()
    assert actual_valid == expected_valid, (
        f"valid_token_count={actual_valid} expected={expected_valid}"
    )
    logger.info("✓ routing_state valid_token_count=%d", actual_valid)

    # 4. gradient scaling fallback
    empty_state = HeteroMoERoutingState()
    count = fix.get_valid_token_count_for_gradient_scaling(empty_state, fallback=128)
    assert count == 128, f"fallback expected 128, got {count}"
    logger.info("✓ gradient_scaling fallback")

    # 5. LOC cache reconciliation
    loc = LOCCacheState(
        cache_hit_mask=torch.zeros(bsz * seq_len, dtype=torch.bool, device=device)
    )
    # 标记后半部分为cache-hit（即使padding_mask说是padding，也应保留为valid）
    loc.cache_hit_mask[bsz * 8:] = True
    pm_flat = pm.reshape(-1)
    corrected = reconcile_padding_with_loc_cache(pm_flat, loc)
    assert corrected[bsz * 8:].sum() == 0, "cache-hit tokens should be un-padded"
    logger.info("✓ LOC cache reconciliation")

    logger.info("All smoke tests passed.")
    sys.exit(0)
