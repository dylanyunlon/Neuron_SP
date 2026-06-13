# Megatron-LM → Neuron_SP Migration Progress

## 状态
- 最新处理: M1587 (Megatron commit #2050 / 9063)
- 总进度: 143/7156 commits (2.0%)
- 实际代码改动 commits: M1445(partial AC), M1500(SwiGLU)
- 小弟 dispatched: M1461(dist ckpt), M1510(MQA)

## 方法论
- SKIP: merge commits, CI changes, TE-specific, Retro, 视觉模型, 无对应Neuron_SP模块
- APPLY: 算法改动映射到 REAL_GPU_BENCHMARK.py (优化器/模型/训练循环)
- 每个 APPLY commit: 20% DES-LOC 适配 + print 诊断

## 下一批重要 commits (需要实际改代码)
- Megatron Flash Attention 集成
- Megatron Sequence Parallelism (Ulysses)
- Megatron Context Parallelism
- Megatron MoE support
- Megatron Muon optimizer

## 如何继续
```bash
cd /path/to/Neuron_SP
# 查看当前进度
git log --oneline | grep "M15" | tail -5
# 继续从 M1588 开始
# 参考 /home/claude/megatron_pending_commits.txt 第 144+ 行
```

## M1930 — Megatron-LM commit 8efc8de8d: Fix MoE aux loss

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@8efc8de8d (Fix moe aux loss, Zijie Yan, 2023-12-26)

### 修复要点
1. **`moe_utils.py` — `switch_load_balancing_loss_func`**:
   - 签名 `(config, gates, mask)` → `(gates, mask, moe_aux_loss_coeff)`，解除对 config 对象的强依赖
   - `mask.size(1)` → `mask.size(-1)`，兼容任意维度
   - 移除 `assert num_experts == config.num_moe_experts`（画蛇添足）
   - `config.aux_loss_coeff` → `moe_aux_loss_coeff`（参数名一致）

2. **`base_moe_layer.py` — `apply_aux_loss`**:
   - 参数 `indicies`（错别字）→ `indices`
   - 传入 `probs`（softmax 全局分布）替代 `scores`（topk 截断后局部归一化）
   - `MoEAuxLossAutoScaler.apply` 作用于 `indices` 而非 `scores`，梯度钩子位置正确

3. **`base_moe_layer.py` — `ZeroDropTopKRouter.routing()`**:
   - `logits = softmax(...)` → `probs = softmax(...)`，名实相符
   - `apply_aux_loss` 调用改为传 `probs`，返回 `indices`

### 鲁迅式评注（20% 适配注记）
> 旧代码以 scores（残羹）论专家负载（天下饥馑），辅助损失徒有其表；  
> 今改以 probs（全局概率），方使负载均衡之损失名副其实。

### 存档文件
- `archive/patches/M1930_Megatron_8efc8de8d_MoE_aux_loss.patch` — 原始 git show 输出
- `archive/patches/M1930_moe_utils_annotated.py` — 鲁迅注解 + print 诊断版 moe_utils
- `archive/patches/M1930_base_moe_layer_annotated.py` — 鲁迅注解 + print 诊断版 base_moe_layer 片段
- `Megatron-LM` submodule 指针已更新至 `8efc8de8d0fc3c617d955c5d1a59b5f321b7511f`
