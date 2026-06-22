# Claude-10 任务: M581-M595 — PII 检测 Pipeline 集成

Session: Claude-10 (M581-M595) | Base: latest main

## 目标
在数据预处理中集成 PII（个人信息）检测和过滤，确保训练数据合规。

## 步骤

### 1. 研读 BigCode PII pipeline
```bash
ls references/data_pipelines/bigcode_pii/
cat references/data_pipelines/bigcode_pii/ner/pii_redaction/*.py | head -200
```

### 2. 实现轻量级 PII 过滤
- 正则匹配: email, IP, API key, SSH key
- 集成到 data pipeline 的 filter stage
- 支持 replace 模式（用 <PII_EMAIL> 等 token 替换）和 drop 模式

### 3. 验证
- 在 100 个代码文件上测试 PII 检出率
- False positive rate < 5%

## 交付物
- 修改的文件 + commit push
- PII 检测统计写入 desloc_results/phase5/
