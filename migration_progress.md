# Megatron-LM → Neuron_SP Migration Progress

## 状态
- 最新处理: M2130 (Megatron ab77e527c -- Rename original_max_position_embeddings in MLATransformerConfig)
- 上一批: M2110 (Megatron 4ca43093f -- MoE fix for Llama4: moe_apply_probs_on_input)
- 总进度: 146/7156 commits (2.0%)
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
# 参考 /home/claude/megatron_pending_commits.txt 第 14## M2130 -- Megatron-LM commit ab77e527c: Rename original_max_position_embeddings

**日期**: 2026-06-13
**来源**: NVIDIA/Megatron-LM@ab77e527c (3 files, ~40 lines)

### 修改要点
1. **`transformer_config.py` — MLATransformerConfig**:
   - 新增 `original_max_position_embeddings: int = 4096` 字段 (YaRN RoPE真实参考长度)
   - `max_position_embeddings` docstring 更新为 "not used, will be deprecated"
   - `__post_init__` 向后兼容：若旧代码设置 max_position_embeddings!=4096，自动迁移值

2. **`multi_latent_attention.py`**:
   - `YarnRotaryEmbedding(...)` 调用：`original_max_position_embeddings=self.config.original_max_position_embeddings`
   - 不再使用 max_position_embeddings 作为 YaRN 参考长度

3. **`test_multi_latent_attention.py`** (4处):
   - 测试配置: `max_position_embeddings=N` → `original_max_position_embeddings=N`

### NSP适配 (REAL_GPU_BENCHMARK.py)
- `TrainingConfig` 新增 `original_max_position_embeddings: int = 4096`
- `get_transformer_config()` 注入 `xformer_cfg['original_max_position_embeddings']`
- `[M2130-XFMR]` print诊断: 打印两个字段值，可在log确认语义分离正确

### 鲁迅式评注（20% 适配注记）
> max_position_embeddings占着位子，实为闲职；original才是主角，却藏身幕后。
> 世人皆用max_position传original之值，如以假名行正事，错乱由此而生。
> 今日正名，立original_max_position_embeddings为显职，
> 令max_position退居deprecation，不失其位，只失其用。

### 存档文件
- `archive/patches/M2130_Megatron_ab77e527c_Rename_original_max_position_embeddings.patch`
- `REAL_GPU_BENCHMARK.py` — TrainingConfig + get_transformer_config() 适配

## M2110 -- Megatron-LM commit 4ca43093f: MoE fix for Llama4

**日期**: 2026-06-13
**来源**: NVIDIA/Megatron-LM@4ca43093f (MoE fix for Llama4, 3 files, 129 lines)

### 修改要点
1. **`experts.py` (GroupedMLP / TEGroupedMLP / SequentialMLP)**:
   - 新增 `moe_apply_probs_on_input` 路径：在 expert MLP forward 之前将路由概率
     乘入 hidden states，再将 probs reset 为 ones（topk=1 强制断言）。
   - 对应 NSP: `moe_topk_router_forward()` stub 的 moe_apply_probs_on_input 分支。

2. **`moe_utils.py` — `topk_softmax_with_capacity` sigmoid 分支**:
   - fp32 upcast（稳定性修复）。
   - expert_bias 在 topk 之前加入（非之后）。
   - use_pre_softmax=True: sigmoid→topk；False: topk→sigmoid（Llama4 默认）。
   - 对应 NSP: `moe_topk_router_forward()` 全路由逻辑。

3. **`transformer_config.py`**:
   - 新字段 `moe_apply_probs_on_input: bool = False`。
   - `moe_router_pre_softmax` docstring 更新（含 sigmoid 说明）。
   - 对应 NSP: `TrainingConfig.moe_apply_probs_on_input / moe_router_pre_softmax / moe_router_topk`。

### 鲁迅式评注（20% 适配注记）
> 旧代码以 sigmoid 先行，再筛 topk，宛如先定名单再考试；
> 新代码以 logit 先筛，再施 sigmoid，方是先考试再发证书的正道。
> 至于 moe_apply_probs_on_input，将路由权重提前吸入隐态，
> 如同酱油入锅早晚之别，味道天壤，但凡 Llama4 之厨，皆知此理。

### 存档文件
- `archive/patches/M2110_Megatron_4ca43093f_MoE_fix_Llama4.patch` -- 原始 Megatron diff 说明 (129行)
- `REAL_GPU_BENCHMARK.py` -- `TrainingConfig` 新增3字段 + `moe_topk_router_forward()` stub

## M2040 -- Megatron-LM commit acba19cb9: Reduce CPU overhead of TEDotProductAttention for packed sequence

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@acba19cb9 (Reduce CPU overhead of TEDotProductAttention for packed sequence)

### 修改要点
1. **`transformer_engine.py` → `REAL_GPU_BENCHMARK.py` `CausalSelfAttention.__init__`**:
   - 原来每次 `forward()` 都调 `get_te_version()` + `.pop()` 过滤 packed_seq_kwargs
   - 现在在 `__init__` 里一次性构建 `_te_packed_fields: frozenset`，把版本 discard 提前
   - `forward()` 直接用 `self._te_packed_fields` 过滤，零 Python 函数调用开销

2. **`CausalSelfAttention.forward()` SDPA 路径**:
   - `[ATTN-SDPA]` print 追加 `[M2040] packed_fields=N (no per-step version checks)` 诊断
   - 可在 log 里确认每层的 packed_fields 数量，验证 TE 版本适配正确

### 鲁迅式评注（20% 适配注记）
> 铁屋中的每一步 forward，都曾悄悄推开版本检查这扇门，再悄悄带走几个微秒。  
> 众人浑然不觉，以为那不过是"查一查而已"——  
> 殊不知千步之行，积弊成山。  
> 今将门锁提前配好，forward 但凭旧钥，再不须当门而立。

### 存档文件
- `archive/patches/M2040_Megatron_acba19cb9_packed_seq_cpu_overhead.patch` — 原始 Megatron diff (60行)
- `REAL_GPU_BENCHMARK.py` — `CausalSelfAttention.__init__` 新增 `_te_packed_fields` 初始化块

4+ 行
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

## M2010 — Megatron-LM commit 96f5c4165: Fix inference pipelining error

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@96f5c4165 (Fix inference pipelining error)

### 核心问题

推理 pipeline 并行时，旧代码以 `is_first_step` 布尔标记区分"首次前向（prompt 阶段）"与"续生成（token-by-token 阶段）"。然而各 pipeline stage 独立为每层分配 KV cache，`is_first_step` 仅反映**本层**是否首次分配，而非**全局时间步**。在多 stage 流水中，此标记对非首层 stage 永远为 `False`，导致：
1. 因果 mask 在 prompt 阶段被错误关闭（应保留）。
2. RoPE 位置编码切片错误——续生成时只应取当前 token 的位置，而非全 prefix。

### 修复要点

1. **删除 `is_first_step`** 布尔标记及其赋值（共 2 处）。
2. **mask 关闭判据改为 `inference_params.sequence_len_offset > 0`**：offset > 0 意味着 prompt forward_step 已完成，处于逐 token 生成阶段，可安全关闭因果 mask。
3. **RoPE 切片重构**：
   - `q_pos_emb[sequence_start:sequence_end, :, :, :]`（统一用 offset 区间，无需 is_first_step 分支）。
   - `k_pos_emb[:sequence_end, :, :, :]`（保持完整历史）。
4. **RoPE 移至 KV cache 写入之前**，先切片再写入，语义更清晰。

### 鲁迅式评注（20% 适配注记）
> is_first_step 如旧官印，印于本层，却不知天下大势；  
> sequence_len_offset 方是朝廷公告，令出必行，各层皆知。  
> 推理流水犹如驿站接力，各驿不可各立旗帜；当以全局偏移为令，方能一令贯通。

### 诊断 print 标记
- `[M2010] inference pipeline fix active — sequence_len_offset-based mask logic` — 每次前向
- `[M2010] sequence_start=... sequence_end=... batch_start=... batch_end=...` — 推理时
- `[M2010] past prompt step (offset=...): mask disabled` — offset > 0 时
- `[M2010] RoPE sliced: q_pos_emb[S:E] k_pos_emb[:E]` — 有 RoPE 时

### 存档文件
- `deepspeed/compile/core_transformer_parallel_attention.py` — 更新（推理 KV cache + RoPE + mask 逻辑重构）

## M2030 — Megatron-LM commit f76b465e0: Add TP communication bootstrap backend interface

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@f76b465e0 (Add TP communication bootstrap backend interface)

### 核心问题

旧代码在 `_initialize_tp_communicators()` 中硬编码 `backend='mpi'`，调用 `torch.distributed.new_group(backend='mpi')`，完全不给用户选择。TransformerEngine >= 1.9.0 已支持在 `initialize_ub()` 内部自建进程组，并接受 `bootstrap_backend` 参数（nccl/mpi/gloo）。旧代码对此毫不知情，仍强制走 MPI 路径。

### 修复要点

1. **`model_parallel_config.py` → `core_base_config.py`**：
   - 新增字段 `tp_comm_bootstrap_backend: str = 'nccl'`，位于 TP comm overlap 字段组末尾。

2. **`arguments.py` → `megatron_arguments.py`**：
   - 新增 `patch_tp_comm_bootstrap_backend_args(parser)`，注册 `--tp-comm-bootstrap-backend` 参数。
   - choices=['nccl', 'mpi', 'gloo']，default='nccl'。

3. **`initialize.py` → `megatron_initialize.py`**：
   - 新增 `initialize_tp_communicators_m2030(te_module, args, ub_cfgs)` 函数。
   - TE >= 1.9.0：直接传 `bootstrap_backend` 给 `initialize_ub()`，TE 内部建组。
   - TE < 1.9.0：若 backend != 'mpi' 则 warnings.warn，然后仍走旧路（`new_group(backend='mpi')` + 无 bootstrap_backend 的 `initialize_ub()`）。

### 鲁迅式评注（20% 适配注记）
> 旧代码以 mpi 为唯一后端，如铁屋中人，虽安于一隅却不知外有天地；  
> nccl 者速，mpi 者稳，gloo 者广——三门洞开，用者自择。  
> TE 版本之墙，以 is_te_min_version 轻轻一推，新旧两路各行其道。  
> 所谓接口，不在于写了多少代码，而在于给了用者多少选择。

### 诊断 print 标记
- `[M2030] core_base_config: tp_comm_bootstrap_backend field added to BaseConfig (default=nccl)` — 模块加载时
- `[M2030] patch_tp_comm_bootstrap_backend_args: --tp-comm-bootstrap-backend registered, default=nccl, choices=[nccl, mpi, gloo]` — 参数注册时
- `[M2030-TP-INIT] initialize_tp_communicators: backend=... input_shape=... tp_size=...` — 初始化入口
- `[M2030-TP-INIT] TE >= 1.9.0 path: passing bootstrap_backend=... to initialize_ub` — 新版 TE 路径
- `[M2030-TP-INIT] TE < 1.9.0 path: creating MPI process group, bootstrap_backend forced to mpi` — 旧版 TE 路径
- `[M2030-TP-INIT] initialize_tp_communicators done.` — 初始化完成

### 存档文件
- `archive/patches/M2030_Megatron_f76b465e0_tp_comm_bootstrap_backend.patch` — 原始 Megatron diff
- `deepspeed/compile/core_base_config.py` — tp_comm_bootstrap_backend 字段
- `deepspeed/compile/megatron_arguments.py` — patch_tp_comm_bootstrap_backend_args()
- `deepspeed/compile/megatron_initialize.py` — initialize_tp_communicators_m2030()

## M2050 — Megatron-LM commit 40fb590e4: Move get_batch_on_this_cp_rank to mcore utils

**日期**: 2026-06-13  
**来源**: NVIDIA/Megatron-LM@40fb590e4

### 修改要点
1. **`core_utils.py`** — 新增 `get_batch_on_this_cp_rank(batch)`:
   - 从 `megatron/training/utils.py` 迁入 `megatron/core/utils.py`（项目映射：`deepspeed/compile/core_utils.py`）
   - 旧版依赖 `get_args().context_parallel_size` + `mpu.get_context_parallel_rank()`
   - 新版改用 `parallel_state.get_context_parallel_world_size()` + `parallel_state.get_context_parallel_rank()`
   - 使 mcore 模块自洽，不再仰赖 training 层

2. **`core_parallel_state.py`** — 新增 CP state 存根:
   - `get_context_parallel_world_size()` → 返回 1（单 GPU safe）
   - `get_context_parallel_rank()` → 返回 0（单 GPU safe）

### 鲁迅式评注（20% 适配注记）
> 鲁迅云：「从来如此，便对么？」旧函数久居 training/utils，依赖 get_args() 与 mpu，如寄人篱下；  
> 今迁 mcore utils，改用 parallel_state API，方为正途。  
> get_args() 与 mpu 之依赖尽除，使 mcore 模块自洽，不必仰赖 training 层之鼻息。

### 诊断 print 标记
- `[M2050] get_context_parallel_world_size: returning 1 (stub)` — CP state 存根调用
- `[M2050] get_context_parallel_rank: returning 0 (stub)` — CP state 存根调用
- `[M2050] get_batch_on_this_cp_rank: cp_size=... keys=[...]` — 入口
- `[M2050] get_batch_on_this_cp_rank: cp_rank=..., slicing sequence dim` — CP>1 时切片路径

### 存档文件
- `deepspeed/compile/core_utils.py` — 新增 get_batch_on_this_cp_rank()
- `deepspeed/compile/core_parallel_state.py` — 新增 get_context_parallel_{world_size,rank}() 存根
- `archive/patches/M2050_Megatron_40fb590e4_cp_rank_to_mcore_utils.patch` — 原始 Megatron diff
