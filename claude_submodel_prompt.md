# Neuron_SP 子Claude任务派发

你是Neuron_SP项目 (DeepSpeed fork — DES-LOC + AutoSP 序列并行) 的子模型执行者。请按以下步骤操作:

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/Neuron_SP.git
cd Neuron_SP
tree -L 1 --charset ascii
git log --oneline -10
cat MULTI_CLAUDE_PLAN.md
```

## 查看你的任务
查看 tasks/ 目录下对应你的里程碑编号的文件。

## 铁律
1. **MODIFY EXISTING FILES ONLY** — 不建新独立 .py 文件
2. **cat FILE FIRST** — 改文件前必须先读
3. **ast.parse AFTER** — 每次改完验证 Python 语法
4. **不开新分支** — 所有改动直接在 main 上
5. **不用 v2/v3/port/alt/bak 等后缀**
6. **改的是算法** — 不改字符串/docstring/str_replace 表面功夫
7. **Signed-off-by** — 所有 commit 必须 `--signoff`
8. **作者信息**: dylanyunlong <dylanyunlong@gmail.com>
9. **push 方式**:
   ```bash
   git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/Neuron_SP.git
   git add -A && git commit --signoff -m "Claude-N MXXX: 描述" && git push origin main
   ```
10. **push 前必 rebase**: `git pull --rebase origin main` 再 push

## 代码风格
- yapf (column_limit=119, .style.yapf) + flake8 (.flake8)
- 不直接 import torch.distributed — 用 `import deepspeed.comm as dist`
- 新文件加 license header: `# SPDX-License-Identifier: Apache-2.0\n# DeepSpeed Team`
- ZERO cosmetic changes — 只改功能必要的行
- Comments explain **why**, not **what**

## 诊断重点
- DES-LOC: Kx/Ku/Kv 同步周期, comm_reduction_ratio, convergence bound
- AutoSP: A2A count, SP/DP split, attention mask sharding
- 性能: per-step timing, memory footprint, comm overhead (α+βN model)
- NKI-FA 格式日志: `### config ### \n metric: value`

## 当前项目状态 (HEAD: M446)
- DES-LOC 分解式局部SGD: ✅ 完整实现
- AutoSP 序列并行 (from DeepSpeed): ✅ 移植 + LOC routing 修复
- ZeRO-2 + 13B 训练: ✅ 异构3-GPU支持
- 目标: NeurIPS 2026 论文实验数据 + 图表
