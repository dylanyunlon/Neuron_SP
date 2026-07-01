# Neuron_SP deepspeed/core/ 架构设计 — V2 (2026-07-01)

## 原则

1. **不按 Megatron commit 镜像**。按子系统划分，每个子系统读 Megatron 对应模块的完整演化历史。
2. **所有模块必须被 import 和调用**。成功标准不是文件存在，而是 `run_pretrain.py` → `desloc_engine.py` → 子系统的 import chain 通畅。
3. **接口先于实现**。本文档定义 class 签名和 import 合约，实现者不得改变签名。

## 子系统划分 (5 个模块，5 个小弟)

### 模块 A: `deepspeed/core/optimizer/distrib_optimizer.py`
- **上游**: `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`
- **职责**: ZeRO-like distributed optimizer，param shard across DP ranks
- **被调用方**: `desloc_engine.py` 的 optimizer 初始化
- **关键类**: `DistributedOptimizer(MegatronOptimizer)`
- **异构适配**: 按 available VRAM（不是 total VRAM）分配 shard 大小

### 模块 B: `deepspeed/core/pipeline_parallel/schedules.py`
- **上游**: `Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
- **职责**: 1F1B / interleaved 1F1B pipeline schedule
- **被调用方**: `desloc_engine.py` 的 train loop
- **关键函数**: `forward_backward_pipelining_with_interleaving()`, `forward_backward_no_pipelining()`
- **异构适配**: 支持不等 stage time（A6000 vs H100）

### 模块 C: `deepspeed/core/distributed/finalize_model_grads.py`
- **上游**: `Megatron-LM/megatron/core/distributed/finalize_model_grads.py`
- **职责**: gradient allreduce/reduce-scatter across DP/TP/PP groups
- **被调用方**: `desloc_engine.py` 的 backward 后 grad sync
- **关键函数**: `finalize_model_grads()`
- **已有代码**: 已经有初版迁移，需要检查和完善

### 模块 D: `deepspeed/core/distributed/param_and_grad_buffer.py`
- **上游**: `Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py`
- **职责**: contiguous parameter/gradient buffers for efficient communication
- **被调用方**: `distrib_optimizer.py` 和 `finalize_model_grads.py`
- **关键类**: `ParamAndGradBuffer`, `Bucket`

### 模块 E: `deepspeed/core/transformer/` (transformer_config + transformer_layer + transformer_block)
- **上游**: `Megatron-LM/megatron/core/transformer/`
- **职责**: TransformerConfig, TransformerLayer, TransformerBlock
- **被调用方**: `run_pretrain.py` 的 LlamaModel 通过 GPTModel → TransformerBlock
- **已有代码**: 有初版迁移但可能不完整

## Import 合约

```
run_pretrain.py
  ├── from deepspeed.core.models import GPTModel                    [E]
  ├── from deepspeed.core.transformer import TransformerConfig       [E]
  ├── from deepspeed.core.tensor_parallel.random import model_parallel_cuda_manual_seed
  └── deepspeed.runtime.desloc_engine.DesLocEngine
        ├── from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer  [A]
        ├── from deepspeed.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving  [B]
        ├── from deepspeed.core.distributed.finalize_model_grads import finalize_model_grads  [C]
        ├── from deepspeed.core.distributed.param_and_grad_buffer import ParamAndGradBuffer  [D]
        └── from deepspeed.core.hetero_bridge import install  [existing]
```

## 派发规则

每个小弟收到的 prompt:
1. `git clone https://github.com/dylanyunlon/Neuron_SP.git`
2. 读 Megatron-LM 的对应源码 (已含在 repo 的 `Megatron-LM/` 子目录)
3. 读已有的 `deepspeed/core/` 对应文件
4. 实现/完善模块，确保 import chain 通畅
5. 输出完整修改后的文件内容

**禁止**: commentary, progress report, 新分支, v2/v3 后缀, 一个 commit 只改几十行
