# M1930: Megatron-LM commit 8efc8de8d — MoE aux loss fix
# 鲁迅注解迁移版本，含 print 诊断，供调试训练时使用
# 对应原文件: megatron/core/transformer/moe/moe_utils.py

import torch

# 鲁迅曰：旧接口以 config 对象为饵，实则暗藏 aux_loss_coeff 与 num_moe_experts 两柄利剑，
# 令调用者须持整只 config 方能入门，殊不知此乃"多余的束缚"。
# 今解其缚，以 moe_aux_loss_coeff 直接传入，一物归一物，不再拖泥带水。
# 旧代码之病：
#   (1) mask.size(1) 强假定 2D，脆弱如纸
#   (2) assert config.num_moe_experts — 断言藏于函数深处，报错时令人摸不着头脑
#   (3) config.aux_loss_coeff 拼写与外部 moe_aux_loss_coeff 不一致，两张皮
# —— M1930 迁移注记，见 Megatron-LM commit 8efc8de8d


def switch_load_balancing_loss_func(gates, mask, moe_aux_loss_coeff):
    """Calculate the auxiliary loss for better load balancing.
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        gates (torch.Tensor): softmax 概率张量（probs），注意非 topk 截断后的 scores。
            旧代码以 scores 喂之，如以残羹论天下饥馑，负载统计失真。
        mask (torch.Tensor): one-hot 专家选择掩码，形状 (tokens, num_experts)。
        moe_aux_loss_coeff (float): 辅助损失系数，直接传入，不再依赖 config 对象。

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    # size(-1) 兼容任意维度，去掉了画蛇添足的 assert
    num_experts = mask.size(-1)
    gates_mean = gates.mean(dim=0)
    selection_mean = mask.float().mean(dim=0)
    aux_loss = torch.sum(gates_mean * selection_mean) * num_experts
    aux_loss *= moe_aux_loss_coeff

    # [诊断 print] 训练初期可开启，确认负载是否均衡、系数是否生效
    print(
        f"[MoE aux_loss diag] num_experts={num_experts}, "
        f"gates_mean={gates_mean.detach().tolist()}, "
        f"selection_mean={selection_mean.detach().tolist()}, "
        f"aux_loss={aux_loss.item():.6f}, coeff={moe_aux_loss_coeff}"
    )

    return aux_loss


def z_loss_func(logits):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The z-loss value.
    """
    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))
    return z_loss
