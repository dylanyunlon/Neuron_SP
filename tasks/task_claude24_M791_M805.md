# Claude-24 任务: M791-M805 — GLM-130B 多任务预训练格式适配

Session: Claude-24 (M791-M805) | Base: latest main

## 目标
参考 GLM-130B 的多任务预训练 (span corruption + causal LM)，为 DES-LOC 添加混合训练模式。

## 参考 (远程)
```bash
git clone --depth 1 https://github.com/THUDM/GLM-130B.git /tmp/glm130b
cat /tmp/glm130b/pretrain_glm.py | head -200
```

## 步骤

### 1. 分析 GLM 的 bidirectional + causal 混合
- GLM 将 70% 数据做 span corruption (BERT-style), 30% 做 causal LM (GPT-style)
- 训练目标混合: `loss = alpha * span_loss + (1-alpha) * causal_loss`

### 2. 在 DES-LOC 的 data pipeline 中添加混合采样
- 修改 `datasets/bigcode/load_commits.py`:
  - commit diff → causal LM (自然的因果结构: before → after)
  - commit message → span corruption (从 diff 中 mask 片段，预测 message)
- 异构适配: 两种 loss 的梯度在 all-reduce 时权重不同

### 3. 验证
- 用 CommitPackFT 的 Python subset 做 smoke test
- 混合 loss 收敛曲线

## 交付物
- 修改的文件 + commit push
