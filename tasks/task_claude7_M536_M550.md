# Claude-7 任务: M536-M550 — BigCode Commit Pipeline 适配 DES-LOC 数据格式

Session: Claude-7 (M536-M550) | Base: latest main

## 目标
将 BigCode 的 GitHub commit 过滤 pipeline 适配为 DES-LOC 预训练数据格式。

## 步骤

### 1. 研读参考代码
```bash
cd Neuron_SP
cat references/data_pipelines/bigcode_commit_filtering/filtering_git_commits.ipynb | python3 -c "
import json,sys
nb=json.load(sys.stdin)
for c in nb['cells']:
    if c['cell_type']=='code': print(''.join(c['source'])[:300]); print('---')
"
```

### 2. 在 deepspeed/runtime/data/ 下扩展现有 data_pipeline
- 添加 commit diff 格式的 tokenizer 适配 (CODE_BEFORE / COMMIT_MSG / CODE_AFTER)
- 复用现有 sequence packing 逻辑
- 集成 quality filter (commit message 黑名单 + diff 范围限制)

### 3. 验证
- `python -c "import ast; ast.parse(open('deepspeed/runtime/data/data_pipeline.py').read())"` 通过
- 单元测试: 10 条 mock commit 数据 → 正确 tokenize + pack

## 交付物
- 修改的文件 + commit push
- 在 desloc_results/phase5/ 下写入测试结果 JSON
