# Claude-8 任务: M551-M565 — MinHash 去重集成到预训练数据 Pipeline

Session: Claude-8 (M551-M565) | Base: latest main

## 目标
将 BigCode 的 MinHash LSH 近似去重移植到 DES-LOC 的数据预处理流程。

## 步骤

### 1. 研读参考
```bash
cat references/data_pipelines/bigcode_dedup/*.py | head -200
```

### 2. 实现
- 在现有 data pipeline 中添加 dedup stage
- 使用 5-gram + Jaccard similarity 0.7 阈值
- 支持增量去重（新数据 vs 已有数据集）
- 异构集群适配: 去重计算可以 offload 到 CPU (1.5TB RAM 充足)

### 3. 基准测试
- 在 1GB 样本数据上测试去重率和速度
- 对比有/无去重对 loss 的影响（如果有预训练 checkpoint）

## 交付物
- 修改的文件 + commit push
- dedup 统计报告写入 desloc_results/phase5/
