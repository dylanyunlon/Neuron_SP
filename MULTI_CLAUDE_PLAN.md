# Neuron_SP Multi-Claude 开发计划

## 协同协议 (Coordination Protocol) — 强制, 所有 Claude(含子Claude)遵守

> 背景: 多个 Claude 并发改 main 会撞车。以下规则消除它。
> (协议从 walpurgis-WTFGG 项目实战验证后移植)

1. **里程碑认领 (claim-before-work)**: 动手前先在下方任务表把你的里程碑行改成
   `🔒 认领中 (Claude-N, 起始HEAD=<短hash>, UTC时间)`, 并**单独先 push 这一行**。
   认领冲突 (同号已被 🔒) → 顺延到下一个空号, 不要并行做同一号。
2. **push 前必 rebase**: `git pull --rebase origin main` 再 push。失败别 `--force`, 解冲突后重试。
3. **一个里程碑 = 一个聚焦改动**: 不在一个里程碑里塞跨模块大改, 减少撞面。
4. **数据完整性闸门**: 任何写进论文/结果文件的数字, **必须**有同仓库已提交的原始凭证。
   禁止 `"attached by user"` 手工转录。未达目标就如实写未达。
5. **分工正交**: 算法改动 (子Claude) / 服务器实跑 (仅 ags1 执行人) / 论文填数 (子Claude, 受闸门4约束)
   三类互不重叠; 服务器任务**不要**派给 hk.cn 子Claude (它够不到 GPU)。
6. **子Claude 派发用简洁第一轮 clone prompt** (见 claude_submodel_prompt.md), 不发大段文字;
   截断时发 `Continue` 续传 (CONV_ID 复用)。

---

## 项目概况

**Neuron_SP** = DeepSpeed fork, 核心贡献:
- **DES-LOC** (分解式局部SGD): 将 U/V/K 参数分组, 以不同周期 Kx/Ku/Kv 同步, 大幅降低通信
- **AutoSP** (自动序列并行): 从 DeepSpeed upstream 移植, 加 LOC routing 适配
- 目标: **NeurIPS 2026** 论文

### 服务器 (ags1)
- 2x A6000 (48GB) + 1x H100 NVL (96GB), AMD EPYC 9354 128核, ~1.5TB RAM
- CUDA 11.5, Driver 550.144
- 路径: /data/jiacheng/system/cache/temp/nips2026/Neuron_SP

### 铁律
- **不开新分支**, 所有改动直接 push 到 main
- **MODIFY EXISTING FILES ONLY** — 不建新独立 .py 文件
- **改的是算法** — 不改字符串/docstring/str_replace 表面功夫
- **Signed-off-by** — 所有 commit 必须 `--signoff`
- 实验结果写入 desloc_results/ 并 push

---

## 已完成的里程碑 (M001-M446)

| 阶段 | 里程碑 | 角色 | 状态 |
|------|--------|------|------|
| 基础设施 | M001-M091 | 项目搭建, DES-LOC 核心实现 | ✅ |
| DES-LOC 完善 | M092-M196 | 通信模型, NKI-FA 日志, 图表 pipeline | ✅ |
| AutoSP 移植 | M257-M331 | 从 DeepSpeed 移植序列并行 + 编译 pass | ✅ |
| 13B 训练 | M332-M446 | ZeRO-2 异构3-GPU, compile 调试, A2A 修复 | ✅ |

### 当前 HEAD: M446 (4fd5cf44)
- strip all try/except降级 — errors must propagate and print

---

## Phase 4: NeurIPS 实验闭环 (M447-M600)

### 第一位 Claude (M447-M460) — 待派发
**角色**: 服务器 13B 完整训练 + 基线对比
任务:
1. git pull origin main 获取所有修复
2. 运行 DES-LOC 13B: `bash run_13B_ags1.sh`
3. 运行 DDP baseline (Kx=Ku=Kv=1)
4. 运行 LocalAdam baseline
5. 结果写入 desloc_results/phase4/
6. 任务文件: tasks/task_claude1_M447_M460.md

### 第二位 Claude (M461-M475) — 待派发
**角色**: Kx sweep 消融实验
任务:
1. Kx ∈ {1, 2, 4, 8, 16, 32} 完整 sweep
2. 固定 Ku=Kv=Kx (同步) 和 Ku=2Kx, Kv=4Kx (分解)
3. 记录: loss curve, comm volume, wall-clock time
4. 任务文件: tasks/task_claude2_M461_M475.md

### 第三位 Claude (M476-M490) — 待派发
**角色**: AutoSP 实验
任务:
1. SP_size ∈ {1, 2, 3} 在 3-GPU 异构环境
2. 测量 A2A overhead vs attention compute
3. SP + DES-LOC 联合实验
4. 任务文件: tasks/task_claude3_M476_M490.md

### 第四位 Claude (M491-M505) — 待派发
**角色**: 多种子评估 + 统计显著性
任务:
1. seed ∈ {42, 123, 456, 789, 1024}
2. mean±std 汇总
3. paired t-test vs DDP baseline
4. 任务文件: tasks/task_claude4_M491_M505.md

### 第五位 Claude (M506-M520) — 待派发
**角色**: NeurIPS 图表生成
任务:
1. Figure 1: Loss vs Step curves (DDP / LocalAdam / DES-LOC)
2. Figure 2: Comm reduction bars
3. Figure 3: Kx sensitivity
4. Figure 4: SP scalability
5. 遵循 NKI-FA draw_plot.py 风格
6. 任务文件: tasks/task_claude5_M506_M520.md

### 第六位 Claude (M521-M535) — 待派发
**角色**: 论文数据填充
任务:
1. 从 desloc_results/ 提取最佳数据
2. 填入 FAUST_nips2026/ 论文表格
3. SOTA 对比表
4. 消融表
5. 任务文件: tasks/task_claude6_M521_M535.md

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| deepspeed/runtime/engine.py | 训练引擎 (DES-LOC 集成) |
| deepspeed/runtime/utils.py | DES-LOC 工具函数 (comm model, log parser) |
| deepspeed/runtime/config.py | DES-LOC 配置 + 图表规格 |
| deepspeed/compile/passes/sp_compile.py | AutoSP 编译 pass |
| deepspeed/compile/init_sp.py | SP 初始化 + GQA fallback |
| deepspeed/compile/custom_ops/all_to_all.py | A2A 通信算子 |
| REAL_GPU_BENCHMARK.py | GPU 基准测试 + 图表生成 |
| run_13B_ags1.sh | ags1 服务器 13B 训练脚本 |
| run_experiment_ags1.sh | ags1 标准实验脚本 |

---

## 派发指南

```bash
# 在主控 Claude 中派发子任务
bash dispatch_all.sh 1      # 派发第1号子Claude
bash dispatch_all.sh all     # 顺序派发所有

# 在 ags1 服务器上运行
cd /data/jiacheng/system/cache/temp/nips2026/Neuron_SP
git pull origin main
bash run_13B_ags1.sh

# 续传 (截断后)
CONV_ID=<uuid> bash claude_hk_chat.sh "Continue"
```

---

## Phase 5: 数据 Pipeline + 框架对标 + 论文完善 (M536-M685)

> 招聘 10 个子 Claude，分 3 条线并行推进

### 数据线 (Claude 7/8/10/12)
| Claude | 里程碑 | 任务 | 依赖 | 状态 |
|--------|--------|------|------|------|
| Claude-7 | M536-M550 | BigCode commit pipeline → DES-LOC 数据格式 | 无 | 🟡 待派发 |
| Claude-8 | M551-M565 | MinHash 去重集成 | Claude-7 | 🟡 待派发 |
| Claude-10 | M581-M595 | PII 检测 pipeline | Claude-7 | 🟡 待派发 |
| Claude-12 | M611-M625 | GHArchive event 提取器 | 无 | 🟡 待派发 |

### 框架线 (Claude 9/11/14/15)
| Claude | 里程碑 | 任务 | 依赖 | 状态 |
|--------|--------|------|------|------|
| Claude-9 | M566-M580 | Megatron PP 调度器对标 → 异构 bubble 优化 | 无 | 🟡 待派发 |
| Claude-11 | M596-M610 | 异构 Tensor Parallel 跨设备分片 | 无 | 🟡 待派发 |
| Claude-14 | M641-M655 | 异构 Memory Profiler | 无 | 🟡 待派发 |
| Claude-15 | M656-M670 | 2022 框架 benchmark 复现 | ags1 服务器 | 🟡 待派发 |

### 论文线 (Claude 13/16)
| Claude | 里程碑 | 任务 | 依赖 | 状态 |
|--------|--------|------|------|------|
| Claude-13 | M626-M640 | DES-LOC 收敛性理论证明 | 无 | 🟡 待派发 |
| Claude-16 | M671-M685 | NeurIPS 论文 Related Work + Experiment | Phase 4 数据 | 🟡 待派发 |

### 依赖图 (可并行执行的)
```
独立启动 (第一批, 同时派发):
  Claude-7  (commit pipeline)     ─┬→ Claude-8  (dedup)
  Claude-9  (PP调度)               │  Claude-10 (PII)
  Claude-11 (异构TP)               │
  Claude-12 (GHArchive)            │
  Claude-13 (理论证明)             │
  Claude-14 (Memory profiler)      │
                                    │
第二批 (依赖第一批):              │
  Claude-15 (benchmark, 需服务器)  │
  Claude-16 (论文, 需 Phase 4 数据)│
```

---

## Phase 6: 预训练框架集成 + 数据管线 (M686-M835)

### 参考框架已拉取
- `references/pretrain_frameworks/gpt_neox/` — EleutherAI GPT-NeoX-20B
- `references/pretrain_frameworks/metaseq_opt/` — Meta OPT-175B
- `references/pretrain_frameworks/bloom_megatron_ds/` — BigScience BLOOM-176B
- 完整索引: `references/pretrain_frameworks/FRAMEWORK_INDEX.md`

### 数据集 (4 个 BigCode commit 数据集)
- StarCoder commits (32GB) / CommitPack (4TB) / CommitPackFT (2GB) / The Stack v2
- 详见: `datasets/bigcode/DATASETS.md`

### 10 位 Sub-Claude 任务分配

| Claude | 里程碑 | 任务 | 参考源 | 状态 |
|--------|--------|------|--------|------|
| Claude-17 | M686-M700 | Rotary Embedding 迁移到 DES-LOC Pipeline | GPT-NeoX | ⬜ 待认领 |
| Claude-18 | M701-M715 | FSDP→ZeRO-3 异构分片适配 | Metaseq OPT | ⬜ 待认领 |
| Claude-19 | M716-M730 | ALiBi 位置编码集成 | BLOOM Megatron-DS | ⬜ 待认领 |
| Claude-20 | M731-M745 | StarCoder Commit Sequence Packing 优化 | BigCode data | ⬜ 待认领 |
| Claude-21 | M746-M760 | CommitPack 4TB 流式加载器 | BigCode data | ⬜ 待认领 |
| Claude-22 | M761-M775 | The Stack v2 PR/Commit 格式适配 | BigCode data | ⬜ 待认领 |
| Claude-23 | M776-M790 | ColossalAI Gemini 异构内存管理 | ColossalAI | ⬜ 待认领 |
| Claude-24 | M791-M805 | GLM-130B 多任务预训练格式 | GLM-130B | ⬜ 待认领 |
| Claude-25 | M806-M820 | Chinchilla Scaling Law 异构验证实验 | DeepMind | ⬜ 待认领 |
| Claude-26 | M821-M835 | CodeGen 代码生成评估框架 | Salesforce | ⬜ 待认领 |

### 任务依赖关系
```
[独立] Claude-17 (Rotary) ──┐
[独立] Claude-19 (ALiBi)  ──┴──→ A/B 对比 (需两者都完成)
[独立] Claude-18 (ZeRO异构) ────→ 基础设施
[顺序] Claude-20 (Packing) → Claude-21 (Streaming) → Claude-22 (Stack v2)
[独立] Claude-23 (Gemini内存)
[独立] Claude-24 (GLM多任务)
[依赖全部] Claude-25 (Scaling Law) — 需要训练基础设施 + 数据管线就绪
[依赖模型] Claude-26 (Eval) — 需要至少一个模型训练完成
```
