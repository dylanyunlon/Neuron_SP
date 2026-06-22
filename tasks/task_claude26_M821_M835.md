# Claude-26 任务: M821-M835 — CodeGen 多轮代码生成评估框架

Session: Claude-26 (M821-M835) | Base: latest main

## 目标
参考 Salesforce CodeGen 的多轮代码生成评估，为 DES-LOC 训练出的模型搭建 eval 框架。

## 参考 (远程)
```bash
git clone --depth 1 https://github.com/salesforce/CodeGen.git /tmp/codegen
cat /tmp/codegen/jaxformer/hf/sample.py | head -100
# 评估: HumanEval, MBPP, MTPB (Multi-Turn Programming Benchmark)
```

## 步骤

### 1. 搭建 eval harness
- 集成 EleutherAI lm-evaluation-harness 作为基础
- 添加 code-specific benchmarks:
  - HumanEval (pass@k)
  - MBPP (basic programming)
  - commit message prediction (我们的特色: 给 diff 预测 message)

### 2. commit message prediction benchmark
- 从 CommitPackFT 中抽取 1000 条测试样本
- 输入: code diff → 输出: commit message
- 评估指标: BLEU, ROUGE-L, exact match rate

### 3. 在 DES-LOC 训练过程中的 periodic eval
- 每 N 个 training step 跑一次 eval
- 异构适配: eval 只在 H100 上跑 (算力最强)，其他 GPU 继续训练下一 epoch 的数据准备

## 交付物
- eval 脚本 + 配置 push
- 在 desloc_results/phase6/ 下写入 eval 框架说明
