# Neuron_SP Architecture: Megatron Core Integration

## 问题

之前的迁移方式是错误的:每个Claude独立生成hetero_xxx.py文件,产出150K行死代码,没有被任何训练路径import。

## 正确做法

不是"把Megatron的commit一个个翻译成新文件",而是:**读懂Megatron模块的完整演化,然后改写deepspeed/core/下已有的骨架文件,使其能被run_pretrain.py和desloc_engine.py实际调用**。

## 当前训练路径 (必须保持工作)

```
run_pretrain.py
  → models/llama_pretrain.py (LlamaModel)
  → deepspeed/runtime/desloc_engine.py (DesLocEngine)
    → torch.distributed (直接使用)
    → torch.optim.AdamW
    → ZeRO-3 sharding (self-built)
    → DES-LOC Kx/Ku/Kv sync
```

## 目标训练路径 (迁移后)

```
run_pretrain.py
  → deepspeed.core.models.GPTModel (替代手写LlamaModel)
  → deepspeed.core.parallel_state (替代直接torch.distributed)
  → deepspeed.core.distributed.DistributedDataParallel (替代手写DDP)
  → deepspeed.core.distributed.finalize_model_grads (替代手写grad sync)
  → deepspeed.core.optimizer.DistributedOptimizer (替代手写ZeRO-3)
  → deepspeed/runtime/desloc_engine.py (保留DES-LOC,但调用core接口)
```

## 模块分工 (7个任务包)

### Task A: parallel_state (1文件)
**文件**: `deepspeed/core/parallel_state.py`
**Megatron源**: `Megatron-LM/megatron/core/parallel_state.py` (2238行)
**职责**: TP/PP/DP/SP/CP process group 初始化和查询
**DES-LOC扩展**: tier group (按GPU型号分组)
**接入点**: desloc_engine.py 的 `dist.new_group()` 调用全部替换为 `parallel_state.get_xxx_group()`

### Task B: distributed (3文件)
**文件**:
- `deepspeed/core/distributed/__init__.py` → `distributed_data_parallel.py`
- `deepspeed/core/distributed/finalize_model_grads.py` (新建)
- `deepspeed/core/distributed/param_and_grad_buffer.py` (新建)

**Megatron源**:
- `Megatron-LM/megatron/core/distributed/distributed_data_parallel.py` (635行)
- `Megatron-LM/megatron/core/distributed/finalize_model_grads.py` (566行)
- `Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py` (按需)

**职责**: DDP wrapper + 梯度同步 + finalize_model_grads
**DES-LOC扩展**: Kx/Ku/Kv 条件同步嵌入 finalize_model_grads
**接入点**: desloc_engine.py 的手写 broadcast/allreduce 替换

### Task C: optimizer (1文件)
**文件**: `deepspeed/core/optimizer/__init__.py` → `distrib_optimizer.py`
**Megatron源**: `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py` (3046行)
**职责**: 分布式optimizer,param shard + grad shard + optimizer state shard
**DES-LOC扩展**: 异构shard sizing (H100拿大份,A6000拿小份)
**接入点**: 替代 desloc_engine.py 的 `self.param_shard_state` + 手写ZeRO-3

### Task D: transformer (4文件)
**文件**:
- `deepspeed/core/transformer/transformer_config.py`
- `deepspeed/core/transformer/transformer_layer.py`
- `deepspeed/core/transformer/transformer_block.py`
- `deepspeed/core/transformer/attention.py` + `mlp.py`

**Megatron源**: `Megatron-LM/megatron/core/transformer/` (全目录)
**职责**: Transformer层定义,兼容TE和原生PyTorch
**DES-LOC扩展**: per-layer tier assignment (哪些层放H100,哪些放A6000)
**接入点**: 替代 `models/llama_pretrain.py` 的手写LlamaModel

### Task E: pipeline_parallel (2文件)
**文件**:
- `deepspeed/core/pipeline_parallel/schedules.py`
- `deepspeed/core/pipeline_parallel/p2p_communication.py`

**Megatron源**:
- `Megatron-LM/megatron/core/pipeline_parallel/schedules.py` (2462行)
- `Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py`

**职责**: 1F1B, interleaved 1F1B pipeline schedule
**DES-LOC扩展**: 异构bubble填充(H100算得快,少等)
**接入点**: desloc_engine.py 的训练循环中可选PP模式

### Task F: tensor_parallel (1文件)
**文件**: `deepspeed/core/tensor_parallel/__init__.py` → 拆分为独立文件
**Megatron源**: `Megatron-LM/megatron/core/tensor_parallel/layers.py`
**职责**: ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
**接入点**: GPTModel 的线性层

### Task G: wiring (修改现有文件,不新建)
**文件**:
- `run_pretrain.py` (修改import路径)
- `deepspeed/runtime/desloc_engine.py` (调用core接口)

**职责**: 把上面A-F的模块接入训练路径
**前置**: A-F全部完成

## 关键约束

1. **不新建hetero_xxx.py** — 改现有文件
2. **每个Task对应Megatron的完整commit历史** — 不是单个commit
3. **必须能被训练代码import** — 不是自嗨的死代码
4. **保持当前训练能跑** — desloc_engine.py的现有功能不能break
5. **文件拆分按Megatron的结构** — 不要把1000行塞进__init__.py
