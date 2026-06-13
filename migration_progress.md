# Megatron-LM → Neuron_SP Migration Progress

## 状态
- 最新处理: M2000 (Megatron de16089be — Distributed optimizer for TE/Apex-independent training)
- 总进度: 145/7156 commits (2.0%)
- 实际代码改动 commits: M1445(partial AC), M1500(SwiGLU)
- 小弟 dispatched: M1461(dist ckpt), M1510(MQA)

## 方法论
- SKIP: merge commits, CI changes, TE-specific, Retro, 视觉模型, 无对应Neuron_SP模块
- APPLY: 算法改动映射到 REAL_GPU_BENCHMARK.py (优化器/模型/训练循环)
- 每个 APPLY commit: 20% DES-LOC 适配 + print 诊断

## 下一批重要 commits (需要实际改代码)
- Megatron Flash Attention 集成
- Megatron Sequence Parallelism (Ulysses)
- Megatron Context Parallelism
- Megatron MoE support
- Megatron Muon optimizer

## 如何继续
```bash
cd /path/to/Neuron_SP
# 查看当前进度
git log --oneline | grep "M15" | tail -5
# 继续从 M1588 开始
# 参考 /home/claude/megatron_pending_commits.txt 第 144+ 行
```

## M1980 — Megatron-LM commit 5486c69c6: Add debug timing utilities

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@5486c69c627e98530dbc556e5c404fed2258b311 (Add debug times, Mikołaj Błaż, 2024-04-09)

### 修改要点
1. **`fully_parallel.py` → `two_stage.py`**:
   - `FullyParallelLoadStrategyWrapper.load()` 内 inline `start/end = time()` → 迁移为 `_exchange_loaded_tensors` 内 `_t0/_t1` 计时
   - 新增 `load_time_total` / `broadcast_time_total` 累计计时器，逐 tensor 统计
   - 新增 `torch.cuda.synchronize()` 计时块（cpu_transfer=False 路径）
   - 替换裸 `print(f'Applying parallel load...')` → `logger.debug` + `print([M1980]...)`
   - `exchange_loaded_tensors_gather_nccl` 的 per-dtype 计时 → 沿用现有 `@timed()` 装饰器（two_stage 无 all_gather 分发轮次）

### 鲁迅式评注（20% 适配注记）
> 铁屋中的 rank，各自负重，唯计时方知谁轻谁重。  
> 旧代码以 print 喧嚷，今改以 logger 静候；  
> 待尘埃落定，cuda.synchronize 一声，再论快慢。

### 存档文件
- `archive/patches/M1980_Megatron_5486c69c6_debug_timing.patch` — 原始 Megatron diff
- `deepspeed/core/dist_checkpointing/strategies/two_stage.py` — 迁移改动（_exchange_loaded_tensors 内联计时）

## M1930 — Megatron-LM commit 8efc8de8d: Fix MoE aux loss

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@8efc8de8d (Fix moe aux loss, Zijie Yan, 2023-12-26)

### 修复要点
1. **`moe_utils.py` — `switch_load_balancing_loss_func`**:
   - 签名 `(config, gates, mask)` → `(gates, mask, moe_aux_loss_coeff)`，解除对 config 对象的强依赖
   - `mask.size(1)` → `mask.size(-1)`，兼容任意维度
   - 移除 `assert num_experts == config.num_moe_experts`（画蛇添足）
   - `config.aux_loss_coeff` → `moe_aux_loss_coeff`（参数名一致）

2. **`base_moe_layer.py` — `apply_aux_loss`**:
   - 参数 `indicies`（错别字）→ `indices`
   - 传入 `probs`（softmax 全局分布）替代 `scores`（topk 截断后局部归一化）
   - `MoEAuxLossAutoScaler.apply` 作用于 `indices` 而非 `scores`，梯度钩子位置正确

3. **`base_moe_layer.py` — `ZeroDropTopKRouter.routing()`**:
   - `logits = softmax(...)` → `probs = softmax(...)`，名实相符
   - `apply_aux_loss` 调用改为传 `probs`，返回 `indices`

### 鲁迅式评注（20% 适配注记）
> 旧代码以 scores（残羹）论专家负载（天下饥馑），辅助损失徒有其表；  
> 今改以 probs（全局概率），方使负载均衡之损失名副其实。

### 存档文件
- `archive/patches/M1930_Megatron_8efc8de8d_MoE_aux_loss.patch` — 原始 git show 输出
- `archive/patches/M1930_moe_utils_annotated.py` — 鲁迅注解 + print 诊断版 moe_utils
- `archive/patches/M1930_base_moe_layer_annotated.py` — 鲁迅注解 + print 诊断版 base_moe_layer 片段
- `Megatron-LM` submodule 指针已更新至 `8efc8de8d0fc3c617d955c5d1a59b5f321b7511f`

## M1960 — Megatron-LM commit c3079ce98: Enable DGRAD RS overlap

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@c3079ce98 (Enable DGRAD RS overlap)

### 修改要点
1. **`model_parallel_config.py` → `core_base_config.py`**:
   - 新字段 `tp_comm_overlap_rs_dgrad: bool = False`
   - 允许 DGRAD GEMM 与 Reduce-Scatter 流水线重叠

2. **`transformer_engine.py` → `core_transformer_custom_layers_transformer_engine.py`**:
   - `TELayerNormColumnParallelLinear.__init__`: TE > 1.6.0.dev0 时向 TE 传入
     `ub_overlap_rs_dgrad=config.tp_comm_overlap_rs_dgrad`
   - 版本守卫与 getattr 安全访问（hasattr 模式）

3. **`arguments.py` → `megatron_arguments.py`**:
   - `patch_tp_comm_rs_dgrad_args(parser)` 注册 `--tp-comm-overlap-rs-dgrad` 标志
   - `dest='tp_comm_overlap_rs_dgrad'`

### 鲁迅式评注（20% 适配注记）
> 梯度反传之际，通信与计算本可并行而行，旧代码却令其相互等待，  
> 犹如官僚衙门，拖沓误事；今以 ub_overlap_rs_dgrad 打通流水，  
> Reduce-Scatter 掩于 DGRAD 之下，性能方得名副其实。

### 诊断 print 标记
- `[M1960]` — 模块加载时打印（3 个文件各一处）
- `[M1960] TELayerNormColumnParallelLinear.__init__ ub_overlap_rs_dgrad=...` — 运行时

## M1990 — Megatron-LM commit 38722c39b: Support jit.script with cross entropy fusion

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@38722c39b (Support jit.script with cross entropy fusion)

### 核心问题

`jit.script`（即 `@jit_fuser`）不允许在被 script 的函数内部调用 `torch.distributed` state 查询函数（`get_tensor_model_parallel_rank` 等）。旧版 `calculate_predicted_logits` 在函数体内做此调用，导致 JIT 编译失败。

### 修复要点

1. **`cross_entropy.py` → `core_tensor_parallel_cross_entropy.py`**:
   - 重构为 `VocabParallelCrossEntropy` 静态方法容器（`calculate_logits_max`、`calculate_predicted_logits`、`calculate_cross_entropy_loss`、`prepare_gradient_calculation_operands`、`calculate_gradients`）。
   - `calculate_predicted_logits` 签名：`vocab_start_index: int`、`vocab_end_index: int` 升格为显式参数。
   - vocab 范围计算移入 `_VocabParallelCrossEntropy.forward()`（非 JIT 上下文），以 int 传入静态方法。
   - `prepare_gradient_calculation_operands` 参数列表末尾加逗号（PEP 8 风格）。

2. **`fused_cross_entropy.py` → `core_fusions_fused_cross_entropy.py`**（**NEW FILE**）:
   - `@jit_fuser` → `@torch.jit.script`（项目映射）。
   - `calculate_predicted_logits`：+`vocab_start_index: int`、+`vocab_end_index: int` 参数。
   - `_VocabParallelCrossEntropy.forward()`：在调用 fused 函数前计算 vocab 范围并传入。
   - `grad_input.bfloat16()` → `grad_input.to(torch.bfloat16)`（jit.script 兼容写法）。
   - `VocabUtility` import 从 `core_tensor_parallel_utils` 引入。

### 鲁迅式评注（20% 适配注记）
> 旧代码在 script 内问询分布式，如入无人之境却偏要拦门验证；  
> 今将问询移出 script，令纯计算之函数得以自由驰骋。  
> vocab 范围本是 int 二数，何必混入张量之 JIT？提而传之，各司其职。

### 诊断 print 标记
- `[M1990] core_tensor_parallel_cross_entropy: jit.script-compatible vocab-range refactor loaded` — 模块加载
- `[M1990] core_fusions_fused_cross_entropy: jit.script cross-entropy fusion loaded` — 模块加载
- `[M1990] _VocabParallelCrossEntropy.forward vocab_start=... vocab_end=...` — 运行时（unfused 路径）
- `[M1990] fused forward vocab_start=... vocab_end=... partition_vocab_size=...` — 运行时（fused 路径）

### 存档文件
- `deepspeed/compile/core_tensor_parallel_cross_entropy.py` — 更新（含 VocabParallelCrossEntropy 类）
- `deepspeed/compile/core_fusions_fused_cross_entropy.py` — 新增 fused 路径
## M2000 — Megatron-LM commit de16089be: Distributed optimizer for TE/Apex-independent training

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@de16089be (Distributed optimizer for TE/Apex-independent training)

### 修改要点
1. **`megatron_optimizer.py` — `HAVE_APEX_OR_TE` 标志**:
   - 新增三级 try/except 导入检测: transformer_engine → apex.optimizers.FusedAdam → torch.optim.Adam
   - `HAVE_APEX_OR_TE=False` 时使用原生 PyTorch Adam，不再强制依赖 Apex/TE
   - 移除 `DistributedOptimizer.__init__()` 中的 `assert HAVE_APEX_OR_TE`（对应 Float16DistributedOptimizer）

2. **`megatron_optimizer.py` — `Float16DistributedOptimizer.state_dict()`**:
   - 缓存 `inner_state_dict = self.optimizer.state_dict()` 避免二次调用
   - `HAVE_APEX_OR_TE=False` 时从 per-param state 提取统一 step 值写入 param_group
   - 旧代码: `{k: v for k, v in self.optimizer.state_dict().items() if k != "state"}` → 新: 复用 inner_state_dict

3. **`megatron_optimizer.py` — `Float16DistributedOptimizer.load_state_dict()`**:
   - `HAVE_APEX_OR_TE=False` 时从 ckpt param_group 读取 step，注入回 per-param optimizer state
   - 原生 Adam 要求 state["step"] 为 float tensor，`torch.tensor(step, dtype=torch.float)`

4. **`megatron_checkpointing.py` — `load_optimizer_state_safe()`**:
   - 新增辅助函数实现 `except KeyError as e: raise e` 模式（替代旧的 sys.exit）
   - 保留完整 traceback，让调用方可以 catch 并做优雅恢复

### 鲁迅式评注（20% 适配注记）
> 铁屋中的 Apex，今已成一道非必要的门槛。  
> 旧代码以 assert HAVE_APEX_OR_TE 把守，寒冬无处投身；  
> 今改以 HAVE_APEX_OR_TE 标志，使原生 Adam 亦可上堂。  
> step 之一字，Apex 藏之于暗，PyTorch 晒之于阳；  
> 保存时摘下，加载时戴回，往来之间，无痛而化。

### 存档文件
- `archive/patches/M2000_Megatron_de16089be_distrib_opt_no_apex.patch` — 原始 Megatron diff
- `deepspeed/compile/megatron_optimizer.py` — HAVE_APEX_OR_TE 标志 + state_dict/load_state_dict 适配
- `deepspeed/compile/megatron_checkpointing.py` — load_optimizer_state_safe() helper (except KeyError as e)
