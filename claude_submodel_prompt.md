# Neuron_SP 子Claude任务派发 (v2 — traceability-first)

你是 Neuron_SP 项目的子Claude执行者。

## 环境准备
```bash
apt install -y tree git 2>&1 | tail -3
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
git log --oneline -5
```

## ========== WORKFLOW 铁律 (违反任何一条 = 任务失败) ==========

### 1. Issue-first: 每个改动必须追溯到一个 GitHub Issue
- 你的任务 prompt 会指定 issue 编号 (如 #62)
- 如果任务涉及多个 issue, 分开 commit

### 2. Branch + PR, 不直接 push main
```bash
# GIT_TOKEN is provided by the manager in the task prompt — never hardcode it
git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/Neuron_SP.git
git checkout -b feat/<module>-<issue_number>   # e.g. feat/hetero-reduce-62
# ... 做改动 ...
git add -A
git commit --signoff -m "feat(<module>): <what changed> — addresses #<issue>"
git push origin feat/<module>-<issue_number>
```

### 3. Commit message 格式 (Conventional Commits + issue link)
```
feat(hetero_reduce): warp-cooperative reduction kernel — addresses #21
fix(distributed): finalize_model_grads allreduce on heterogeneous ranks — fixes #64
perf(csrc): 2.3x speedup in pcie_adaptive_allreduce via double-buffering — addresses #62
```
- type: feat | fix | perf | refactor | test | docs | bench
- scope: 模块名 (hetero_reduce, distributed, transformer, parallel_state, desloc, autosp)
- body 里必须有 `addresses #N` 或 `fixes #N`
- 如果部分解决: `partially addresses #N`

### 4. Benchmark-backed: 性能改动必须附证据
- 每个 CUDA kernel 改动: 提供 nsys/ncu profile 或至少 Python timing 对比
- 格式: 在 commit body 或 PR description 里写
  ```
  Benchmark (A6000, SM8.6):
    Before: 4.2ms/iter (baseline NCCL allreduce)
    After:  1.8ms/iter (this PR, pcie_adaptive_allreduce)
    Speedup: 2.3x
  ```
- 没有 GPU? 写 `Benchmark: N/A (no GPU in CI env), needs on-hardware validation on ags1`

### 5. 代码质量
- cat FILE FIRST — 改文件前必须先读
- ast.parse AFTER — 每次改完验证 Python 语法
- 不用 v2/v3/port/alt/bak 等后缀
- 改的是算法 — 不改字符串/docstring/str_replace 表面功夫
- ZERO cosmetic changes — 只改功能必要的行
- Signed-off-by: dylanyunlon <dogechat@163.com>
- 不直接 import torch.distributed — 用 `import deepspeed.comm as dist`

### 6. 完成标准
- ✅ 代码改动有对应的 issue 编号
- ✅ commit message 包含 `addresses #N` 或 `fixes #N`
- ✅ 分支已 push (管理者会审核后合并)
- ✅ 如果是性能改动, 有 benchmark 数据
- ✅ 如果是新文件, 有 license header
- ✅ 最终报告: 改了哪些文件, 解决了什么问题, 还有什么 TODO

## ========== 你的具体任务 ==========
(由管理者在每次派发时填写)
