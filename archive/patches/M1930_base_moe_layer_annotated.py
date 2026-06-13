# M1930: Megatron-LM commit 8efc8de8d — MoE aux loss fix
# 鲁迅注解迁移版本，含 print 诊断
# 对应原文件: megatron/core/transformer/moe/base_moe_layer.py
# 修改点 1: apply_aux_loss 方法签名 + 逻辑
# 修改点 2: ZeroDropTopKRouter.routing() 变量命名 logits→probs

# ────────────── apply_aux_loss (修改点 1) ──────────────

def apply_aux_loss(self, loss_func, probs, indices):
    """Apply auxiliary loss for load balancing.

    鲁迅曰：旧代码以 scores（topk 后归一化）喂给损失函数，
    却对"probs 才是全局概率分布"视而不见，
    犹如以残羹论天下饥馑——看似在统计，实则数字已经失真。
    改动要点：
      - 参数名 indicies（错别字）→ indices（正名）
      - scores → probs，确保损失函数看到真实的 softmax 全局分布
      - MoEAuxLossAutoScaler 作用对象由 scores 改为 indices，
        梯度钩子挂到路由决策处，反向传播才能正确更新路由权重。
    """
    mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)

    # [诊断 print] 检查语义对齐：probs 形状应为 (tokens, num_experts)
    print(
        f"[MoE apply_aux_loss diag] probs.shape={probs.shape}, "
        f"mask.shape={mask.shape}, num_experts={self.num_experts}"
    )

    aux_loss = loss_func(probs, mask, self.config.moe_aux_loss_coeff)

    # AutoScaler 挂 indices，令梯度正确回传至路由决策
    indices = MoEAuxLossAutoScaler.apply(indices, aux_loss)
    return indices


# ────────────── ZeroDropTopKRouter.routing() 关键段 (修改点 2) ──────────────

# 鲁迅曰：旧代码将 softmax(logits) 仍赋值给 logits，
# 以"logits"之名行"probs"之实，如以假面示人，
# 终令后来者在 Z-Loss 与 topk 之间迷失语义。
# 今正其名曰 probs，一字之改，脉络顿清。

probs = torch.softmax(logits, dim=-1)

# [诊断 print] 确认 softmax 输出正常（每行之和应约等于 1.0）
print(
    f"[MoE routing diag] probs.shape={probs.shape}, "
    f"probs.min={probs.min().item():.6f}, "
    f"probs.max={probs.max().item():.6f}, "
    f"probs.sum_per_row(mean)={probs.sum(dim=-1).mean().item():.6f}"
)

if self.config.moe_z_loss_coeff > 0:
    probs = self.apply_z_loss(probs)

scores, indices = torch.topk(probs, k=self.k, dim=1)
scores /= scores.sum(dim=-1, keepdim=True)

if self.config.moe_aux_loss_coeff > 0:
    # 传入 probs（全局），返回带梯度钩子的 indices
    indices = self.apply_aux_loss(self.moe_aux_loss_func, probs, indices)

# scores 保持纯净，不再被 AutoScaler 污染
# return scores, indices
