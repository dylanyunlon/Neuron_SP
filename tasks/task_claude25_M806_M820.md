# Claude-25 任务: M806-M820 — Chinchilla Scaling Law 在异构集群上的验证实验设计

Session: Claude-25 (M806-M820) | Base: latest main

## 目标
设计一套 scaling law 验证实验，在 2×A6000+1×H100 上跑 70M/160M/410M/1B 四个模型规模。

## 参考
```bash
# Chinchilla (Hoffmann et al., 2022): C_opt ≈ 20 × N (最优 tokens ≈ 20× 参数量)
# 但异构集群的有效算力不等于峰值算力之和
cat references/pretrain_frameworks/FRAMEWORK_INDEX.md
```

## 步骤

### 1. 设计实验矩阵
| 模型 | 参数量 | Chinchilla最优Tokens | 预计时间(异构) |
|------|--------|---------------------|---------------|
| Tiny | 70M | 1.4B | ~2小时 |
| Small | 160M | 3.2B | ~8小时 |
| Medium | 410M | 8.2B | ~1天 |
| Large | 1B | 20B | ~5天 |

### 2. 在 DES-LOC 中实现 auto-config
- 修改 `pretrain_7b.py` 或对应的 config:
  - 根据模型大小自动选择: Pipeline stages 数量 / ZeRO stage / batch size
  - 输出: 每个配置的 `tokens_per_second` 和 `MFU`
- 添加 print() 诊断: 实际 vs 理论吞吐

### 3. 拟合 scaling curve
- L(N, D) = E + A/N^α + B/D^β (Chinchilla 公式)
- 在异构集群上，N 和 D 的关系可能偏离标准 Chinchilla
- 记录异构特有的 overhead: pipeline bubble ratio, cross-NUMA latency

## 交付物
- 实验配置文件 push
- 论文 Section 5.x "Heterogeneous Scaling Laws" 的实验框架
