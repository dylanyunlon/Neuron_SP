# ISSUE 1 — PP=5 Pipeline Bubble：1F1B Schedule 启用方案

**问题**: `config.use_pipeline_schedule` 默认 `False`，PP=5 跑的是裸 for loop（`desloc_engine.py` line 2493），  
每个 micro-batch 串行执行 forward+backward，5 个 stage 顺序依赖，bubble rate ≈ (PP−1)/PP = **80%**。

> 注：文中"80%"而非"60%"是正确算法。串行 for-loop 下每个 micro-batch 的有效计算时间  
> = T_fwd + T_bwd，而所有其他 stage 在这段时间全部空闲。对于 PP=5，每步实际只有  
> 1/5 的 cycle 是有效计算，即 bubble rate = 4/5 = 80%。标准 1F1B 将其压缩至  
> (PP−1)/（num_microbatches + PP−1），当 num_mb≥8 时约 33%。

---

## 1. 现状诊断

### 1.1 代码路径

```
desloc_engine.py:2452-2454   get_pipeline_forward_backward(cfg, default_fn=None)
  └── core_adapters.py:86      if not config.use_pipeline_schedule: return default_fn
                                ↑ default_fn=None → _pipeline_fb_func = None

desloc_engine.py:2631          if _cp_fb is not None: ...   ← CP schedule 路径
desloc_engine.py:2645-2661     else: self.forward(...) / scaled_loss.backward()
                                ↑ 实际执行路径：micro 循环内逐个 fwd+bwd
```

`_pipeline_fb_func` 被 `get_pipeline_forward_backward` 返回后，在训练循环中从未被调用——  
当前引擎只有 CP 路径（`_cp_fb`）有接入点，PP 路径的 adapter 存在但没有对应的调用分支。

### 1.2 已有基础设施

- `schedules.py` 已实现完整的 `forward_backward_pipelining_without_interleaving`（非交错 1F1B）
- `P2PCommunicator` 已实现 `send_forward_recv_backward`、`send_backward_recv_forward`、批量 `batch_isend_irecv`
- M3766 异步 P2P send 安全等待已合入（`send_forward_recv_backward` 返回 handle，在 backward 前 wait）
- `set_pipeline_layer_split([4,8,8,4,8])` 已实现，`get_pipeline_model_parallel_rank_for_layer` 已就绪
- `HeterogeneousBubbleFiller` 骨架已存在（`schedules.py:625`）

---

## 2. 拓扑分析：PP=5 异构 NUMA 下 1F1B 可行性

### 2.1 硬件拓扑

```
NUMA 0                        NUMA 1
┌─────────────────────┐       ┌──────────────────┐
│ GPU0 (stage 0, 4层) │       │ GPU3 (stage 3,4层)│
│ GPU1 (stage 1, 8层) │  PCIe │ GPU4 (stage 4,8层)│
│ GPU2 (stage 2, 8层) │  跨越 │                  │
└─────────────────────┘       └──────────────────┘

stage 分配: [4, 8, 8, 4, 8] 共 32 层
层数权重:   [1x, 2x, 2x, 1x, 2x]
```

**跨 NUMA 链路**：stage2→stage3（GPU2→GPU3）和 stage3→stage4（GPU3→GPU4）在 forward 方向均需跨越 PCIe，  
而 stage0↔stage1↔stage2 在 NUMA0 内通过 NVLink/PCIe 互联（具体带宽取决于板卡型号）。

### 2.2 1F1B 能用吗？——**可以，但必须处理两个异构问题**

标准 1F1B 假设**所有 stage 的 forward 时延近似相等**（对称假设）。本拓扑下有两处不对称：

#### 问题 A：层数不均导致 stage 计算时间不同

| Stage | 层数 | 相对计算时间 |
|-------|------|-------------|
| 0     | 4    | 1×          |
| 1     | 8    | 2×          |
| 2     | 8    | 2×          |
| 3     | 4    | 1×          |
| 4     | 8    | 2×          |

**影响**：stage0 和 stage3 计算结束快，会在等待 stage1/4 的激活时产生额外等待。  
**结论**：1F1B 的调度表（warmup 微批数）依赖 rank 编号而非 wall-clock 时间，  
这种等待是 pipeline bubble 的一部分，**1F1B 不会因此破坏正确性**——只是快 stage 的利用率略低。

#### 问题 B：跨 NUMA 通信时延增加 P2P 等待

stage2→stage3 的 `send_forward`/`recv_forward` 经过 PCIe 跨 NUMA，延迟约 5-10µs，  
而 NUMA 内同 PCIe 域约 2-3µs。

**影响**：steady-state 1F1B 的 send_forward+recv_backward 是同步等待的（`send_forward_recv_backward`），  
跨 NUMA 的这两次通信会轻微延长 stage2 和 stage3 的 bubble。  
**结论**：仍然正确，只是 stage2/3 的有效利用率比均匀拓扑低约 5-8%，可通过 M3766 异步 send 部分掩盖。

#### 问题 C：warmup/cooldown 的 activation 内存峰值不均

标准 non-interleaved 1F1B 中，stage `r`（0-indexed）在 warmup 阶段积累 `PP−r−1` 个 micro-batch 的激活。

```
stage 0: warmup 4 microbatches → 4× activation in flight（最多）
stage 1: warmup 3 microbatches → 3× activation
stage 2: warmup 2 microbatches → 2× activation
stage 3: warmup 1 microbatch  → 1× activation
stage 4: warmup 0 microbatches → 0（最后 stage）
```

stage0 存 4 个激活但每个激活只经过 4 层（小），stage1 存 3 个激活但每个经过 8 层（大）。  
**实际内存峰值 ≈ stage1**（3 × 8层激活）> stage0（4 × 4层激活），这是反直觉的，需在内存预算中注意。

**结论：1F1B 在本拓扑下完全可用**，正确性无问题，需要额外处理的仅是内存峰值估算和异步 send 安全。

---

## 3. 启用方案

### 3.1 第一步：补全 desloc_engine.py 的 PP 调用分支

当前引擎在 `for micro in range(num_microbatches)` 内逐 micro-batch 调用 `self.forward()`。  
1F1B schedule 要求**接管整个 num_microbatches 循环**，因此需要在循环**外**调用 `_pipeline_fb_func`。

在 `desloc_engine.py` 的步级循环（`for step in range(...)`）中，修改如下：

```python
# 伪代码 diff，实际位置约 line 2492-2668
if _pipeline_fb_func is not None:
    # --- 1F1B schedule 路径 ---
    # 构造 forward_step_func 适配器（与 _cp_fb 路径对称）
    _fwd_store = _pipeline_fb_func(
        forward_step_func   = _make_forward_step_func(self),
        data_iterator       = self.data_iter,          # 外层 iter，schedule 自行驱动
        model               = self.model,
        num_microbatches    = num_microbatches,
        seq_length          = cfg.seq_length,
        micro_batch_size    = cfg.micro_batch_size,
        forward_only        = False,
        p2p_communicator    = self.p2p_communicator,
        pg_collection       = self._pg_collection,
    )
    step_loss = sum(float(l) for l in _fwd_store) / max(len(_fwd_store), 1)
    if self.fp32_grad_manager is not None:
        self.fp32_grad_manager.accumulate()
else:
    # 原有 for micro in range(num_microbatches) 循环（保持不变）
    ...
```

**注意**：`_pipeline_fb_func` 自身负责所有 micro-batch 的 fwd+bwd+P2P，  
调用方不再需要 `for micro` 循环，原 micro loop 完整保留在 `else` 分支作为回退。

### 3.2 第二步：构造 `_make_forward_step_func`

`forward_backward_pipelining_without_interleaving` 需要 `forward_step_func(data_iter, model) → (loss_tensor, num_tokens_tensor)` 签名。  
需要包装 `self.forward()` 并保留 MoE aux loss、FP32 grad accum 等逻辑：

```python
def _make_forward_step_func(engine):
    def _forward_step(data_iter, model):
        import torch
        batch = next(data_iter)
        input_ids = batch["tokens"] if isinstance(batch, dict) else batch[0]
        labels    = batch.get("labels") if isinstance(batch, dict) else (batch[1] if len(batch) > 1 else None)
        input_ids = input_ids.to(engine._local_dev, non_blocking=True)
        if labels is not None:
            labels = labels.to(engine._local_dev, non_blocking=True)

        loss, scaled_loss = engine.forward(input_ids, labels,
                                            num_microbatches=engine._cur_num_microbatches)
        # MoE aux loss（仅 last stage 计算）
        if engine.moe_adapter is not None:
            _aux = engine.moe_adapter.collect_aux_loss()
            if not isinstance(_aux, float) or _aux != 0.0:
                scaled_loss = scaled_loss + _aux / max(engine._cur_num_microbatches, 1)

        seq_len = input_ids.shape[-1]
        num_tokens = torch.tensor(seq_len, dtype=torch.int64, device=input_ids.device)
        return scaled_loss, num_tokens
    return _forward_step
```

### 3.3 第三步：初始化 P2PCommunicator 和 pg_collection

`forward_backward_pipelining_without_interleaving` 在 `p2p_communicator=None` 时会自动从  
`parallel_state` 构建，但 DES-LOC 使用自定义 PP group（`BridgeCommunicator` 路径），  
需在 engine `__init__` 时构建并存储：

```python
# desloc_engine.py __init__ 末尾（当 use_pipeline_schedule=True）
if cfg.use_pipeline_schedule and cfg.pipeline_parallel_size > 1:
    from deepspeed.core.pipeline_parallel.p2p_communication import P2PCommunicator
    from deepspeed.core.process_groups_config import ProcessGroupCollection
    from deepspeed.core.model_parallel_config import ModelParallelConfig
    import deepspeed.core.parallel_state as ps

    _mp_cfg = ModelParallelConfig(
        pipeline_model_parallel_size=cfg.pipeline_parallel_size,
        tensor_model_parallel_size=getattr(cfg, 'tensor_parallel_size', 1),
        variable_seq_lengths=True,           # 必须 True（跨 NUMA 激活 shape 可能异构）
        deallocate_pipeline_outputs=True,    # 节省跨 NUMA stage 的 activation 驻留内存
    )
    _pp_group = ps.get_pipeline_model_parallel_group()
    self._p2p_comm = P2PCommunicator(pp_group=_pp_group, config=_mp_cfg)

    self._pg_collection = ProcessGroupCollection()
    self._pg_collection.tp     = ps.get_tensor_model_parallel_group()
    self._pg_collection.cp     = ps.get_context_parallel_group()
    self._pg_collection.pp     = _pp_group
    self._pg_collection.dp_cp  = ps.get_data_parallel_group(with_context_parallel=True,
                                                              partial_data_parallel=False)
    self._pg_collection.tp_dp_cp = ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
    self._pg_collection.embd   = ps.get_embedding_group(check_initialized=False)
    self._pg_collection.pos_embd = ps.get_position_embedding_group(check_initialized=False)

    # 注册异构 layer split
    from deepspeed.core.pipeline_parallel.schedules import set_pipeline_layer_split
    set_pipeline_layer_split(cfg.pipeline_layer_split)   # e.g. [4,8,8,4,8]
```

### 3.4 第四步：启用 async P2P send 安全（M3766）

在 `ModelParallelConfig` 中设置 `deallocate_pipeline_outputs=True`（已在上方），  
同时确认 `p2p_communication.py` 的 `send_forward_recv_backward` 在返回前等待 isend handle：

```python
# p2p_communication.py - send_forward_recv_backward 实现已含（M3766 已合入）：
# reqs = batch_isend_irecv([isend(output), irecv(grad)])
# for req in reqs: req.wait()   ← 等待 isend 完成后才允许调用方释放 output_tensor
```

若 `deallocate_pipeline_outputs=True`，schedule 在 backward 之前会调用 `deallocate_output_tensor`，  
此时必须确保 isend 已 wait——M3766 正是解决这个 race condition 的。

### 3.5 第五步：TrainingConfig 新增字段

```python
# desloc_engine.py TrainingConfig dataclass
pipeline_parallel_size: int = 1
"""PP world size. Must match parallel_state initialization."""

pipeline_layer_split: List[int] = field(default_factory=list)
"""Per-stage layer counts for heterogeneous PP.
Example for 5-stage [4,8,8,4,8]: total 32 layers across NUMA0+NUMA1."""
```

### 3.6 最小化 config.yaml 改动

```yaml
use_pipeline_schedule: true
pipeline_parallel_size: 5
pipeline_layer_split: [4, 8, 8, 4, 8]

# 不启用 VPP（virtual_pipeline_model_parallel_size 保持 null）
# → get_forward_backward_func 返回 forward_backward_pipelining_without_interleaving
# → 标准 non-interleaved 1F1B，最简路径

# 推荐同时开启：
# use_bridge_communicator: true   ← BridgeCommunicator for stage2↔stage3 cross-NUMA
# deallocate_pipeline_outputs: true  ← 通过 ModelParallelConfig 传入
```

---

## 4. 异构拓扑下的 Warmup/Cooldown 处理

### 4.1 标准 1F1B warmup 计算（`schedules.py:2054`）

```python
num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
#                         = PP - rank - 1
# stage0: warmup 4, stage1: warmup 3, ..., stage4: warmup 0
```

这个公式与层数无关——只依赖 rank 编号，**对异构拓扑天然正确**。

### 4.2 "快 stage"（stage0/3，4层）的 warmup 细节

stage0 在 warmup 期间连续执行 4 次 forward，每次只经过 4 层，完成后等待来自 stage4  
的第一个 grad（cooldown 开始的信号）。这 4 次 forward 的总时间 ≈ stage1 执行 2 次  
8层 forward 的时间，**时间匹配良好**——stage0 不会因为快而在 warmup 期空等。

stage3（4层）类似：warmup=1，steady-state 期间每轮执行 fwd+bwd 各一个 micro-batch，  
其 fwd 时间是 stage4 的一半。stage3 会有额外空闲，但这是 pipeline bubble 的固有成本，  
1F1B 已将其最小化至 `(PP−1) × t_stage / (num_mb × t_stage) = 4/num_mb`。

### 4.3 不同 stage 速度对 bubble 的影响（量化分析）

设 `t_slow = 2t`（8层 stage），`t_fast = t`（4层 stage），num_microbatches = M。

| schedule | 有效利用率 |
|----------|-----------|
| 裸 for-loop（当前） | 1/PP = **20%** |
| 1F1B 均匀假设       | M/(M+PP−1) ≈ **67%** （M=8） |
| 1F1B 实际异构       | 受限于最慢 stage（stage1/2/4，8层）|

关键：PP schedule 下整个 pipeline 的吞吐受**最慢 stage** 限制（pipeline 定律）。  
stage1（8层）是 stage0（4层）的 2× 慢，因此 stage0 在 steady-state 每轮都会等待约 `t_slow − t_fast = t`。  
这是"层数不均"引起的内生 bubble，只能通过**重新平衡 layer 分配**（如改为 [8,8,8,8,0]  
或引入 interleaved VPP）来消除，而非 1F1B 调度本身的问题。

**建议**：启用 1F1B 后，profile stage wall-clock 时间，若 stage0/3 空闲率 > 30%，  
考虑调整 `pipeline_layer_split` 为更均匀的分配（如 `[6,7,7,6,6]`）。

### 4.4 Cooldown 阶段跨 NUMA 的异步安全

Cooldown 在 `schedules.py:2229` 中逐一 `recv_backward` + `backward_step` + `send_backward`，  
全部是同步操作，**不存在 race condition**，M3766 的 async send 等待在 steady-state 已覆盖。

唯一需要额外关注的是 stage2→stage3 的 `send_backward`：BridgeCommunicator 使用  
`batch_isend_irecv`，跨 NUMA 的 isend 完成后 stage2 才能释放 `input_tensor_grad`。  
确认 `BridgeToP2PWrapper.send_backward` 内部等待 isend 完成（或在调用方 wait handle）。

---

## 5. 实施顺序与风险控制

### Phase 1（最小可用，1-2天）
1. `TrainingConfig` 加 `pipeline_parallel_size` 和 `pipeline_layer_split` 字段
2. 在 `desloc_engine.__init__` 中当 `use_pipeline_schedule=True` 时初始化 `P2PCommunicator`
3. 在训练循环 `for step` 内、`for micro` 外，加 `if _pipeline_fb_func is not None` 分支
4. 实现 `_make_forward_step_func` wrapper，保留 MoE aux loss 路径
5. 配置 `variable_seq_lengths=True` + `deallocate_pipeline_outputs=True`

**验证**：单机 5xGPU，`num_microbatches=8`，`pipeline_layer_split=[4,8,8,4,8]`，  
运行 10 steps，比较 loss 曲线与裸 for-loop 是否一致（loss 应完全相同，schedule 不影响数值）。

### Phase 2（性能分析，1天）
6. 用 `torch.profiler` 采集 step timeline，测量各 stage 的 bubble time
7. 比较 1F1B 与裸 for-loop 的 step time（预期加速 2-3×，取决于 num_microbatches）
8. Profile stage0/3 的空闲率，决定是否需要调整 `pipeline_layer_split`

### Phase 3（可选优化，3-5天）
9. 启用 `HeterogeneousBubbleFiller`：为 stage0/3 填充额外 forward-only micro-batch（gradient checkpointing 的 lookahead）
10. 考虑 interleaved 1F1B（`virtual_pipeline_model_parallel_size=2`）：将 bubble 从 `(PP−1)/M` 进一步降至 `(PP−1)/(M×VPP)`，代价是每 stage 需存储 2× micro-batch 激活

### 回退策略
`use_pipeline_schedule=False`（默认）→ 完整回退到裸 for-loop，  
任何时候都可以一行 config 切换，零风险。

---

## 6. 关键注意事项（异构+跨 NUMA 特有）

### 6.1 collective 对称性

PP schedule 下，每个 stage 调用 `forward_step` 的次数必须**完全一致**（等于 `num_microbatches`），  
否则 ZeRO-3 的 all_gather collective 会 hang（参见 MEGATRON_INSIGHTS.md §1.1 的分析）。  
`forward_backward_pipelining_without_interleaving` 保证了这一点，不需要额外处理。

`hetero_scheduler.schedule()` 返回的 `num_microbatches` 必须在所有 PP ranks 上相同——  
当前 hetero scheduler 允许"per-rank micro_batch_size"差异，但 micro-batch 数量必须统一。  
若 hetero scheduler 可能返回不同数值，需在调用 `_pipeline_fb_func` 前做 `dist.all_reduce(num_microbatches, op=MIN)`。

### 6.2 data iterator 的消耗

裸 for-loop 下，每个 rank 独立调用 `next(self.data_iter)` 共 `num_microbatches` 次。  
1F1B schedule 下，**`forward_step_func` 内部驱动 data iterator**，调用次数仍是 `num_microbatches` 次，  
但位置从 engine 循环移到了 schedule 内部。必须确保 `_make_forward_step_func` 使用的  
iterator 就是 `self.data_iter`（而不是重建的），避免数据偏移。

### 6.3 shard sync wait 的位置

当前代码在 `micro == 0` 时等待 `_shard_sync_stream`（line 2572）。  
1F1B schedule 路径下，`micro` 循环消失，这个 wait 需要移到 `_pipeline_fb_func` 调用之前（step 级别），  
确保 BF16 参数在第一个 micro-batch 的 forward 开始前完全同步。

### 6.4 fp32_grad_manager 的 accumulate 时机

当前在每个 micro-batch backward 后调用 `fp32_grad_manager.accumulate()`（逐步 BF16→FP32 提升）。  
1F1B schedule 将所有 micro-batch 的 backward 内聚在 `_pipeline_fb_func` 内部，  
需要将 `accumulate()` 注册为 `config.grad_sync_func` 的一部分，或在 `finalize_model_grads_func` 中调用。  
最简方案：在 `_pipeline_fb_func` 返回后调用一次 `fp32_grad_manager.accumulate()`（接受轻微精度损失），  
严格方案：通过 `ModelParallelConfig.no_sync_func` + backward hook 在每个 micro-batch 后触发。

---

## 7. 预期收益

| 指标 | 裸 for-loop | 1F1B（num_mb=8） |
|------|-------------|-----------------|
| Bubble rate | ~80% | ~33% |
| Stage 利用率 | ~20% | ~60-67% |
| 步级时间（相对） | 1.0× | **0.35-0.45×** |
| 峰值 activation 内存 | 1×（串行） | 4×（stage0 warmup 4个）|
| 跨 NUMA transfer 次数/step | num_mb × 2 | num_mb × 2（相同） |

激活内存增加是 1F1B 的固有代价（warmup 阶段存多个 micro-batch 的激活），  
对于 stage0（4层，小激活）影响可控；stage1（8层，大激活，warmup=3）需提前确认 VRAM 预算。

---

## 参考

- `deepspeed/core/pipeline_parallel/schedules.py` — 完整 1F1B 实现
- `deepspeed/runtime/core_adapters.py:76` — `get_pipeline_forward_backward` adapter
- `deepspeed/runtime/desloc_engine.py:2452` — `_pipeline_fb_func` 获取点
- `deepspeed/runtime/desloc_engine.py:2493` — 当前裸 for-loop（待替换）
- Megatron M3544 (0ca9b6395) — multimodule pipelining in 1F1B schedule
- Megatron M3766 (260cba713) — async P2P send safety (`deallocate_pipeline_outputs`)
