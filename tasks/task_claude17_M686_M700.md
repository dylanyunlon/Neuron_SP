# Claude-17 任务: M686-M700 — GPT-NeoX Rotary Embedding 迁移到 DES-LOC Pipeline

Session: Claude-17 (M686-M700) | Base: latest main

## 目标
将 EleutherAI GPT-NeoX 的 Rotary Position Embedding 实现迁移到 DES-LOC 异构 pipeline 的每个 stage。

## 参考文件
```bash
cat references/pretrain_frameworks/gpt_neox/transformer.py | grep -A 50 "class RotaryEmbedding"
cat references/pretrain_frameworks/gpt_neox/transformer.py | grep -A 30 "apply_rotary"
```

## 步骤

### 1. 读懂 NeoX Rotary 实现
- NeoX 的 rotary embedding 支持 interleaved 和 non-interleaved 两种模式
- 理解 cos/sin cache 与 sequence position 的关系

### 2. 修改 deepspeed/runtime/pipe/ 下的 pipeline module
- 在每个 stage 的 transformer block 中集成 rotary embedding
- 关键: rotary 的 position offset 在 pipeline 的不同 stage 间必须正确传递
- 异构适配: A6000 用 FP16 rotary, H100 用 BF16 rotary, 精度对齐

### 3. 验证
- `python -c "import ast; ast.parse(open('<修改文件>').read())"` 通过
- position offset 在 2-stage pipeline 下的正确性测试

## 交付物
- 修改的文件 + commit push
- 在 desloc_results/phase6/ 下写入 rotary 精度对比 JSON
