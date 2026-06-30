# ISSUE-5: desloc_engine.py 架构重构方案

**状态**: 设计草稿，待评审  
**负责人**: ArchLead  
**依据**: Megatron-LM `training/` 拆分模式 + DeepSpeed `runtime/` 子目录模式  
**前置**: ISSUE-4 合并后开始  

---

## 1. 背景与问题

`deepspeed/runtime/desloc_engine.py` 当前 **4208 行**，单文件承载了以下完全不同职责：

| 职责 | 行范围 | 行数 |
|---|---|---|
| 枚举/数据类型定义 | 118–197 | 80 |
| `TrainingConfig` 配置 dataclass | 198–451 | 253 |
| `HeteroRegistry`（内联副本） | 452–576 | 124 |
| `TierDiscovery`（硬件拓扑发现） | 577–712 | 135 |
| `PartitionSolver`（分区规划） | 713–970 | 257 |
| `MiniTransformer` 等模型定义 | 971–1142 | 171 |
| `DesLocEngine.__init__`（初始化） | 1165–2432 | 1267 |
| `DesLocEngine.train()`（训练主循环） | 2486–3373 | 887 |
| `DesLocEngine.save/load_checkpoint()` | 3380–4095 | 715 |
| 工具函数和 `_smoke_test` | 4096–4208 | 112 |

**根本问题**：任何人改 checkpoint 逻辑都要打开一个 4208 行的文件，PR diff 无法 review。

同时存在大量 `hetero_*` 文件命名混乱：有的是项目核心功能，有的是零引用死代码，有的应该去掉 `hetero_` 前缀成为正式模块。

参考项目对比：

- **Megatron `megatron/training/`**：`training.py`（编排）、`checkpointing.py`、`initialize.py`、`arguments.py` 各司其职，单文件 ≤ 800 行
- **DeepSpeed `deepspeed/runtime/`**：`checkpoint_engine/`（子目录）、`comm/`（子目录）隔离关注点

---

## 2. hetero_* 文件分析

### 2.1 死代码（ref = 0）——应删除

以下文件在整个 `deepspeed/` 代码库中**零引用**，既无生产调用，也无测试覆盖，属于废弃试验代码：

| 文件 | 行数 | 判定理由 |
|---|---|---|
| `deepspeed/runtime/zero/hetero_fsdp_double_buffer.py` | 1561 | 全库零引用，`zero/` 目录无任何 import |

> **注**：`hetero_fsdp_double_buffer.py` 是唯一真正孤立的零引用文件。其他几个看似低引用的文件（`hetero_memory_manager.py`、`hetero_optimizer_router.py`）在 `zero/stage3.py` 和 `zero/__init__.py` 中有引用，不能删。

**操作**：`git rm deepspeed/runtime/zero/hetero_fsdp_double_buffer.py`

---

### 2.2 仅测试引用（无生产调用）——条件保留或归档

| 文件 | 行数 | 现状 | 建议 |
|---|---|---|---|
| `deepspeed/pipe/hetero_mimo_parallelism.py` | 819 | 仅被 `deepspeed/tests/test_hetero_mimo_parallelism.py` 引用 | 保留但移入 `deepspeed/pipe/mimo_parallelism.py`（去前缀，见 §2.3），测试文件同步改名 |

---

### 2.3 活跃文件——应重命名（去掉 hetero_ 前缀）

以下文件已是项目**核心生产模块**，`hetero_` 前缀是历史遗留的"试验"标记，现在已经是唯一实现，继续使用 `hetero_` 前缀会让新人误以为是可替换的备选实现。

#### runtime/ 下的核心调度/计算模块

| 旧文件名 | 新文件名 | 引用方 | 重命名理由 |
|---|---|---|---|
| `runtime/hetero_step_batch_scheduler.py` | `runtime/step_batch_scheduler.py` | `desloc_engine.py`, `hetero_registry.py`, `dataloader.py` | 项目唯一批量调度器，已是正式功能 |
| `runtime/hetero_grad_norm_skip.py` | `runtime/grad_norm_skip.py` | `desloc_engine.py`, `hetero_registry.py`, `engine.py` | 梯度范数跳过是通用能力，非试验 |
| `runtime/hetero_fp32_grad_accum.py` | `runtime/fp32_grad_accum.py` | `desloc_engine.py`, `hetero_registry.py` | FP32 梯度累积是标准训练组件 |
| `runtime/hetero_gdn_selective_recompute.py` | `runtime/selective_recompute.py` | `desloc_engine.py`, `hetero_registry.py` | 选择性重计算策略，非异构专属 |
| `runtime/hetero_elastic_batch.py` | `runtime/elastic_batch.py` | `desloc_engine.py` | 弹性批量重分配，是正式功能 |
| `runtime/hetero_mimo_training_loop.py` | `runtime/mimo_training_loop.py` | `desloc_engine.py`, `core_adapters.py` | MIMO 训练循环是核心路径 |
| `runtime/hetero_registry.py` | `runtime/extension_registry.py` | `engine.py`, `hetero_mesh.py`, `compile/init_sp.py` | Registry 本身不是异构的，是通用扩展点 |

#### checkpoint/ 下的 checkpoint 子系统

| 旧文件名 | 新文件名 | 引用方 | 重命名理由 |
|---|---|---|---|
| `checkpoint/hetero_checkpoint_config.py` | `checkpoint/checkpoint_config.py` | `desloc_engine.py` | checkpoint 配置，非异构专属 |
| `checkpoint/hetero_async_checkpoint_save.py` | `checkpoint/async_checkpoint_save.py` | `desloc_engine.py` | 异步保存是生产功能 |
| `checkpoint/hetero_async_checkpoint_load.py` | `checkpoint/async_checkpoint_load.py` | `desloc_engine.py` | 异步加载是生产功能 |

#### 其他目录

| 旧文件名 | 新文件名 | 引用方 | 重命名理由 |
|---|---|---|---|
| `compile/custom_ops/hetero_mesh.py` | `compile/custom_ops/device_mesh.py` | `tier_aware_param_layout.py`, `engine.py`, `__init__.py`, `sp_dp_registry.py`, `init_sp.py` | 设备网格是基础设施，非试验 |
| `elasticity/hetero_flextron_config.py` | `elasticity/flextron_config.py` | `elasticity/__init__.py` | Flextron 配置，elasticity 子模块内部 |
| `pipe/hetero_mimo_parallelism.py` | `pipe/mimo_parallelism.py` | 仅测试 | 见 §2.2 |

#### 保持原名（有合理 hetero_ 语义）

| 文件 | 保留理由 |
|---|---|
| `runtime/zero/hetero_memory_manager.py` | 管理异构层级内存，`hetero_` 准确描述其专属职责 |
| `runtime/zero/hetero_optimizer_router.py` | 按 tier 路由优化器，`hetero_` 语义准确 |

---

### 2.4 死代码补充：des_loc_* 孤立文件

以下文件全库零引用，是重构过程中留下的未使用草稿，应随本次重构一并清理：

| 文件 | 行数 | 状态 |
|---|---|---|
| `deepspeed/runtime/des_loc_grad_clipping.py` | 1352 | 全库零引用 |
| `deepspeed/runtime/des_loc_mimo_training_loop.py` | 1637 | 全库零引用 |
| `deepspeed/runtime/desloc_partition.py` | 1281 | 全库零引用（`desloc_engine.py` 内有内联实现） |

**操作**：`git rm deepspeed/runtime/des_loc_*.py deepspeed/runtime/desloc_partition.py`

---

## 3. desloc_engine.py 拆分方案

参考 Megatron `training/` 的层次结构，将 4208 行文件拆为以下模块：

```
deepspeed/runtime/
├── desloc_engine.py          ← 保留：DesLocEngine 类骨架（~600 行后）
├── training_config.py        ← 新建：TrainingConfig dataclass
├── tier_types.py             ← 新建：枚举 + TierSpec + PartitionPlan
├── tier_discovery.py         ← 新建：TierDiscovery + PartitionSolver
├── initialize.py             ← 新建：DesLocEngine.__init__ 的核心初始化逻辑
├── training.py               ← 新建：DesLocEngine.train() 主循环
├── extension_registry.py     ← 重命名自 hetero_registry.py（同时删除内联副本）
└── checkpointing/            ← 新建子目录（对应 DeepSpeed checkpoint_engine/ 模式）
    ├── __init__.py
    ├── config.py             ← 来自 hetero_checkpoint_config.py
    ├── async_save.py         ← 来自 hetero_async_checkpoint_save.py
    └── async_load.py         ← 来自 hetero_async_checkpoint_load.py
```

### 3.1 各文件职责说明

**`tier_types.py`**（~80 行）
- `PartitionStrategy`、`TierClass` 枚举
- `TierSpec` dataclass
- `PartitionPlan` dataclass
- 不依赖任何其他本地模块，是纯数据类型层

**`training_config.py`**（~253 行）
- `TrainingConfig` dataclass 及其所有字段
- `_CHECKPOINT_DIR` 常量
- 依赖 `tier_types.py` 中的 `PartitionStrategy`

**`tier_discovery.py`**（~390 行）
- `TierDiscovery` 类（硬件拓扑探测）
- `PartitionSolver` 类（ZeRO-3 / Pipeline 分区规划）
- `build_warmup_cosine_scheduler()` 工具函数（或移入 `utils.py`）
- `infinite_data_iter()` 工具函数

**`initialize.py`**（~1267 行 → 最终目标 <500 行）
- `DesLocEngine.__init__` 的各阶段初始化提取为私有函数：
  - `_init_distributed(config, tiers)` — 分布式组初始化
  - `_build_model(config, tiers)` — 模型构建与 DDP 包装
  - `_build_optimizer(config, model)` — 优化器和调度器
  - `_init_checkpointing(config)` — checkpoint 配置加载
  - `_init_extensions(engine)` — HeteroRegistry 发现与注册
- 这些函数由 `DesLocEngine.__init__` 依次调用，不改变外部行为

**`training.py`**（~887 行）
- `DesLocEngine.train()` 主循环体
- 提取为独立函数 `run_training_loop(engine)` 或保留为方法但文件独立导入
- logging 后端初始化（W&B / TensorBoard）也在此处

**`checkpointing/`**（子目录，参考 DeepSpeed `checkpoint_engine/` 模式）
- 将 `save_checkpoint()`、`load_checkpoint()`、`_apply_loaded_state()`、`drain_checkpoint()` 合计 ~715 行提取
- 对外通过 `checkpointing/__init__.py` 暴露 `save`, `load` 函数
- `async_save.py` 和 `async_load.py` 的内容来自现有 `hetero_async_*` 文件（重命名后）

**`desloc_engine.py`（重构后）**（目标 ~400 行）
- 只保留 `DesLocEngine` 类骨架
- `__init__`：调用 `initialize.py` 中的各个 `_init_*` 函数
- `train()`：委托 `training.py` 中的 `run_training_loop()`
- `save/load_checkpoint()`：委托 `checkpointing/` 子目录
- `forward()`、`step()`、`close()` 保留（短方法，合计 ~60 行）
- 工具类方法 `get_tier_by_class()`、`memory_summary()` 保留（~25 行）

---

## 4. MiniTransformer 模型定义处理

`desloc_engine.py` 中内嵌了完整的 Transformer 模型定义（`RMSNorm`、`CausalSelfAttention`、`MLP`、`TransformerBlock`、`MiniTransformer`，共 171 行）。这些是 **smoke test / 单元测试用的参考实现**，不应与引擎代码混在一起。

**方案**：移入 `deepspeed/runtime/models/mini_transformer.py`（新建），`_smoke_test()` 函数一并迁移。

---

## 5. 内联 HeteroRegistry 副本问题

`desloc_engine.py` 第 452–576 行有一个 `HeteroRegistry` 类的**内联实现**，与 `hetero_registry.py`（重命名后为 `extension_registry.py`）**功能重复**。

**方案**：
1. 先完成 §2.3 的 `hetero_registry.py` → `extension_registry.py` 重命名
2. 再删除 `desloc_engine.py` 中的内联 `HeteroRegistry` 类
3. `desloc_engine.py` 改为 `from deepspeed.runtime.extension_registry import ExtensionRegistry`

---

## 6. Git Commit 计划（原子提交，逐步可测试）

每个 commit 保持项目可运行（不做半途的改名），按以下顺序执行：

---

### Phase 1 — 清理死代码（无风险，最先做）

**Commit 1a**：`chore: remove zero-ref hetero_fsdp_double_buffer dead code`
```
git rm deepspeed/runtime/zero/hetero_fsdp_double_buffer.py
```
- 全库零引用，无副作用

**Commit 1b**：`chore: remove unreferenced des_loc_* draft files`
```
git rm deepspeed/runtime/des_loc_grad_clipping.py
git rm deepspeed/runtime/des_loc_mimo_training_loop.py
git rm deepspeed/runtime/desloc_partition.py
```
- 三个文件全库零引用，是拆分过程中未接入的草稿

> **Review checkpoint**：CI 通过后再继续，确认删除无副作用。

---

### Phase 2 — hetero_* 重命名（分批，按目录隔离）

**Commit 2a**：`refactor(checkpoint): rename hetero_* checkpoint files to canonical names`
```
git mv deepspeed/checkpoint/hetero_checkpoint_config.py    deepspeed/checkpoint/checkpoint_config.py
git mv deepspeed/checkpoint/hetero_async_checkpoint_save.py deepspeed/checkpoint/async_checkpoint_save.py
git mv deepspeed/checkpoint/hetero_async_checkpoint_load.py deepspeed/checkpoint/async_checkpoint_load.py
```
同步更新 `desloc_engine.py` 中的三处 lazy import 路径。

**Commit 2b**：`refactor(runtime): rename hetero_* scheduler and grad files`
```
git mv deepspeed/runtime/hetero_step_batch_scheduler.py  deepspeed/runtime/step_batch_scheduler.py
git mv deepspeed/runtime/hetero_grad_norm_skip.py        deepspeed/runtime/grad_norm_skip.py
git mv deepspeed/runtime/hetero_fp32_grad_accum.py       deepspeed/runtime/fp32_grad_accum.py
```
同步更新 `desloc_engine.py`、`engine.py`、`dataloader.py`、`hetero_registry.py` 中的 import。

**Commit 2c**：`refactor(runtime): rename hetero_* recompute, elastic_batch, mimo files`
```
git mv deepspeed/runtime/hetero_gdn_selective_recompute.py deepspeed/runtime/selective_recompute.py
git mv deepspeed/runtime/hetero_elastic_batch.py           deepspeed/runtime/elastic_batch.py
git mv deepspeed/runtime/hetero_mimo_training_loop.py      deepspeed/runtime/mimo_training_loop.py
```
同步更新 `desloc_engine.py`、`core_adapters.py` 的 import。

**Commit 2d**：`refactor(runtime): rename hetero_registry to extension_registry`
```
git mv deepspeed/runtime/hetero_registry.py deepspeed/runtime/extension_registry.py
```
同步更新 `engine.py`、`compile/custom_ops/hetero_mesh.py`（此时还未重命名）、`compile/init_sp.py`。

**Commit 2e**：`refactor(compile): rename hetero_mesh to device_mesh`
```
git mv deepspeed/compile/custom_ops/hetero_mesh.py deepspeed/compile/custom_ops/device_mesh.py
```
同步更新 `tier_aware_param_layout.py`、`engine.py`、`compile/custom_ops/__init__.py`、`sp_dp_registry.py`、`init_sp.py`。

**Commit 2f**：`refactor(elasticity): rename hetero_flextron_config to flextron_config`
```
git mv deepspeed/elasticity/hetero_flextron_config.py deepspeed/elasticity/flextron_config.py
```
同步更新 `elasticity/__init__.py`。

**Commit 2g**：`refactor(pipe): rename hetero_mimo_parallelism to mimo_parallelism`
```
git mv deepspeed/pipe/hetero_mimo_parallelism.py deepspeed/pipe/mimo_parallelism.py
git mv deepspeed/tests/test_hetero_mimo_parallelism.py deepspeed/tests/test_mimo_parallelism.py
```
更新测试文件内的 import。

> **Review checkpoint**：Phase 2 完成后所有重命名到位，CI 通过，再开始 Phase 3 拆分。

---

### Phase 3 — desloc_engine.py 拆分（按 Megatron 模式）

**Commit 3a**：`refactor(runtime): extract tier_types.py from desloc_engine`
- 新建 `deepspeed/runtime/tier_types.py`
- 将 `PartitionStrategy`、`TierClass`、`TierSpec`、`PartitionPlan` 移入
- `desloc_engine.py` 顶部改为 `from .tier_types import *`
- **不改任何逻辑**，仅移动代码

**Commit 3b**：`refactor(runtime): extract training_config.py from desloc_engine`
- 新建 `deepspeed/runtime/training_config.py`
- 将 `TrainingConfig` dataclass（含 `_CHECKPOINT_DIR`）移入
- `desloc_engine.py` 改为 `from .training_config import TrainingConfig`

**Commit 3c**：`refactor(runtime): extract tier_discovery.py (TierDiscovery + PartitionSolver)`
- 新建 `deepspeed/runtime/tier_discovery.py`
- 将 `TierDiscovery`、`PartitionSolver`、`build_warmup_cosine_scheduler()`、`infinite_data_iter()` 移入
- `desloc_engine.py` 改为从此模块 import

**Commit 3d**：`refactor(runtime): extract models/mini_transformer.py and _smoke_test`
- 新建 `deepspeed/runtime/models/__init__.py`
- 新建 `deepspeed/runtime/models/mini_transformer.py`
- 将 `RMSNorm`、`CausalSelfAttention`、`MLP`、`TransformerBlock`、`MiniTransformer`、`_smoke_test()` 移入
- `desloc_engine.py` 的 `__init__` 中 lazy import `MiniTransformer` 时从新路径取

**Commit 3e**：`refactor(runtime): remove inline HeteroRegistry duplicate from desloc_engine`
- 删除 `desloc_engine.py` 中 452–576 行的内联 `HeteroRegistry` 类
- 在文件头部 import `from .extension_registry import ExtensionRegistry as HeteroRegistry`（兼容别名，后续再改名）

**Commit 3f**：`refactor(runtime): extract checkpointing/ subpackage from desloc_engine`
- 新建 `deepspeed/runtime/checkpointing/__init__.py`
- 新建 `deepspeed/runtime/checkpointing/engine_checkpointing.py`，内容为 `save_checkpoint`、`load_checkpoint`、`_apply_loaded_state`、`drain_checkpoint` 函数（接受 `engine` 参数）
- 将 `desloc_engine.py` 的四个方法体替换为单行委托调用：
  ```python
  def save_checkpoint(self, path): from .checkpointing import save_checkpoint; save_checkpoint(self, path)
  ```

**Commit 3g**：`refactor(runtime): extract training.py (train loop) from desloc_engine`
- 新建 `deepspeed/runtime/training.py`
- 将 `DesLocEngine.train()` 的 887 行主体提取为 `run_training_loop(engine: DesLocEngine) -> None`
- `desloc_engine.py` 中 `train()` 方法变为：
  ```python
  def train(self) -> None:
      from .training import run_training_loop
      run_training_loop(self)
  ```

**Commit 3h**：`refactor(runtime): extract initialize.py (__init__ phases) from desloc_engine`
- 新建 `deepspeed/runtime/initialize.py`
- 将 `__init__` 内的 5 个初始化阶段各自提取为 `_init_distributed()`、`_build_model()`、`_build_optimizer()`、`_init_checkpointing()`、`_init_extensions()` 函数
- `__init__` 变为这 5 个函数的调用序列（可见 ~100 行）
- **这是最高风险的 commit**，需要最仔细的测试

> **Review checkpoint**：Phase 3 完成后 `desloc_engine.py` 应 ≤ 500 行。跑完整的 smoke_test 和 CI。

---

### Phase 4 — 收尾与文档

**Commit 4a**：`docs: update module docstrings after refactor`
- 更新各新文件的模块级 docstring，说明职责和迁移来源

**Commit 4b**：`docs: add architecture diagram to docs/ISSUE5_architecture_refactor.md`
- 将最终文件树和 import 依赖图补充进本文档（此处留白）

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|---|---|---|
| hetero_* 重命名破坏运行时动态发现（`HeteroRegistry.discover()` 用 `pkgutil.walk_packages` 按前缀扫描） | 中 | Phase 2 完成后专项验证：在 `extension_registry.py` 的 discover() 里临时打印扫描到的模块列表，确认无遗漏 |
| `DesLocEngine.__init__` 拆分后 self 属性初始化顺序依赖 | 高 | Commit 3h 必须保持函数调用顺序与原 `__init__` 完全一致；加 `assert hasattr(self, 'config')` 守卫 |
| `lazy import` dict `self._lazy` 在 `train()` / `save_checkpoint()` 中大量使用 | 中 | 提取前先梳理 `self._lazy` 的所有 key，确保 `training.py` 和 `checkpointing/` 都能访问到 |
| `des_loc_*` 文件删除后发现有外部用户脚本引用 | 低 | 这三个文件全库零引用，git log 确认均未被 tag/release 引用过 |

---

## 8. 执行前提与不做的事

**前提**：
- ISSUE-4 完成，`hetero_elastic_batch` 功能已稳定
- CI 绿灯才进入下一 Phase

**本次不做**：
- 不改任何函数逻辑，只移动代码
- 不改 `zero/hetero_memory_manager.py` 和 `zero/hetero_optimizer_router.py` 的名字（语义准确，保持原名）
- 不合并 checkpoint 子目录到 `deepspeed/checkpoint/`（那是 DeepSpeed 上游的目录；本项目的运行时 checkpoint 逻辑放在 `runtime/checkpointing/` 保持隔离）
- 不在本 PR 里引入任何新功能

---

## 附录 A：文件引用矩阵（完整）

| 文件 | 引用它的文件数 | 状态 |
|---|---|---|
| `checkpoint/hetero_async_checkpoint_load.py` | 1 (desloc_engine) | → `checkpoint/async_checkpoint_load.py` |
| `checkpoint/hetero_async_checkpoint_save.py` | 1 (desloc_engine) | → `checkpoint/async_checkpoint_save.py` |
| `checkpoint/hetero_checkpoint_config.py` | 1 (desloc_engine) | → `checkpoint/checkpoint_config.py` |
| `compile/custom_ops/hetero_mesh.py` | 5 | → `compile/custom_ops/device_mesh.py` |
| `elasticity/hetero_flextron_config.py` | 1 | → `elasticity/flextron_config.py` |
| `pipe/hetero_mimo_parallelism.py` | 1 (test only) | → `pipe/mimo_parallelism.py` |
| `runtime/hetero_elastic_batch.py` | 1 (desloc_engine) | → `runtime/elastic_batch.py` |
| `runtime/hetero_fp32_grad_accum.py` | 2 | → `runtime/fp32_grad_accum.py` |
| `runtime/hetero_gdn_selective_recompute.py` | 2 | → `runtime/selective_recompute.py` |
| `runtime/hetero_grad_norm_skip.py` | 3 | → `runtime/grad_norm_skip.py` |
| `runtime/hetero_mimo_training_loop.py` | 2 | → `runtime/mimo_training_loop.py` |
| `runtime/hetero_registry.py` | 3 | → `runtime/extension_registry.py` |
| `runtime/hetero_step_batch_scheduler.py` | 3 | → `runtime/step_batch_scheduler.py` |
| `runtime/zero/hetero_fsdp_double_buffer.py` | **0** | **DELETE** |
| `runtime/zero/hetero_memory_manager.py` | 1 | 保留原名 |
| `runtime/zero/hetero_optimizer_router.py` | 1 | 保留原名 |
| `runtime/des_loc_grad_clipping.py` | **0** | **DELETE** |
| `runtime/des_loc_mimo_training_loop.py` | **0** | **DELETE** |
| `runtime/desloc_partition.py` | **0** | **DELETE** |
