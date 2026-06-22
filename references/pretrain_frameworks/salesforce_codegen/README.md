# Salesforce CodeGen 参考文件 (Salesforce Research)
上游: github.com/salesforce/CodeGen (2022)
模型: CodeGen-16B (多轮代码生成)

## 文件说明
- `modeling_codegen.py` — CodeGen 模型定义 (GPT-J 架构变体)
- `configuration_codegen.py` — 模型配置
- `train_deepspeed.py` — DeepSpeed 训练脚本 (!)
- `sample.py` — 采样/生成
- `mtpb_sample.py` — Multi-Turn Programming Benchmark 采样
- `mtpb_exec.py` — MTPB 执行评估

## DES-LOC 参考点
- train_deepspeed.py: CodeGen 用 DeepSpeed 训练! 直接可对比
- CodeGen 架构 = GPT-J (rotary embedding), 与 NeoX 同源
- MTPB 多轮评估: commit diff → 代码生成的评估范式
- 代码预训练的数据混合策略 (NL + PL 混合)
