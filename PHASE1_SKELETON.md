# Phase 1: deepspeed/core/ Architecture Skeleton

## Problem
We have 275K lines of Python and only 3.3K lines of our own CUDA C++.
A Megatron-grade heterogeneous training system needs tight C++/CUDA kernels
for communication, fused ops, and tier-aware scheduling. The existing csrc/hetero_reduce/
has 5 kernels but they're not integrated into the training path.

## Architecture: What Each Sub-Claude Builds

### Module A: `deepspeed/core/parallel_state.py` + `deepspeed/core/distributed/`
**Scope**: Full TP/PP/DP/SP/CP group initialization for 5 heterogeneous GPUs.
**Key files**: parallel_state.py, distributed/distributed_data_parallel.py, 
distributed/finalize_model_grads.py, distributed/param_and_grad_buffer.py
**Contract**: `initialize_model_parallel(tp, pp, dp, sp)` → sets all process groups.
Training calls `mpu.get_data_parallel_group()`, `mpu.get_tensor_model_parallel_group()`, etc.
**Integration point**: `run_pretrain.py` calls `initialize_model_parallel()` at startup.

### Module B: `csrc/hetero_reduce/` CUDA kernels + Python op_builder
**Scope**: Production-grade CUDA kernels for heterogeneous training.
**Key files**:
- `csrc/hetero_reduce/hetero_reduce.cu` — tier-aware reduce-scatter
- `csrc/hetero_reduce/fused_swiglu_ln.cu` — fused SwiGLU + LayerNorm  
- `csrc/hetero_reduce/pcie_adaptive_allreduce.cu` — PCIe bandwidth-aware allreduce
- `csrc/hetero_reduce/fused_rope_hetero.cu` — fused RoPE for heterogeneous heads
- `csrc/hetero_reduce/tier_activation_offload.cu` — async activation CPU offload
- NEW: `csrc/hetero_reduce/fused_cross_entropy.cu` — fused CE loss
- NEW: `csrc/hetero_reduce/quantized_comm.cu` — FP8 communication for Blackwell
- `op_builder/hetero_reduce.py` — build config
- `deepspeed/ops/hetero_reduce/hetero_reduce_op.py` — Python wrapper
**Contract**: `from deepspeed.ops.hetero_reduce import HeteroReduceBuilder; op = HeteroReduceBuilder().load()`
**Integration**: `desloc_engine.py` calls fused kernels during gradient sync.

### Module C: `deepspeed/core/transformer/` — Attention + MLP + TransformerLayer
**Scope**: The transformer block with heterogeneous head-count padding, MLA support.
**Key files**: attention.py, dot_product_attention.py, mlp.py, transformer_block.py,
transformer_layer.py, transformer_config.py, multi_latent_attention.py
**Contract**: `TransformerLayer(config, layer_number)` → `.forward(hidden, attention_mask)`
**Integration**: `desloc_engine.py` wraps model that contains TransformerBlock.

### Module D: `deepspeed/core/optimizer/` — Distributed optimizer + tier-aware sharding
**Scope**: ZeRO optimizer with heterogeneous shard sizing (A6000 gets smaller shards).
**Key files**: distrib_optimizer.py, optimizer.py, optimizer_config.py, clip_grads.py
**Contract**: `DistributedOptimizer(optimizer, args)` with `_compute_hetero_shard_boundaries()`
**Integration**: `desloc_engine.py` creates the optimizer via this module.

### Module E: `deepspeed/core/pipeline_parallel/` — PP=5 heterogeneous pipeline
**Scope**: 1F1B schedule for uneven pipeline stages across different GPU tiers.
**Key files**: schedules.py, p2p_communication.py, combined_1f1b.py
**Contract**: `forward_backward_pipelining_without_interleaving(fwd_step, bwd_step, ...)`
**Integration**: Called from training loop when `pp_size > 1`.

### Module F: `deepspeed/core/hetero_bridge/` — The glue layer
**Scope**: Tier discovery, shard planning, engine integration, AutoSP hooks.
**Key files**: tier_map.py, shard_planner.py, engine_integration.py, autosp_hook.py,
dist_opt_adapter.py, pp_schedule_adapter.py, desloc_sync_policy.py
**Contract**: `HeteroBridge.install(engine)` → configures all tier-aware behavior.
**Integration**: `desloc_engine.py` calls `HeteroBridge.install()` during init.

## Dispatch Plan

Each module = 1 sub-Claude conversation. The sub-Claude:
1. Clones the repo in its container
2. Reads ALL commits touching that module (`git log --all -- <paths>`)
3. Reads the Megatron-LM reference implementation
4. Produces a COMPLETE, importable subsystem (not stubs)
5. Verifies with `python -c "from deepspeed.core.X import Y"` 
6. Does NOT push (manager pushes after review)

## API Contract (frozen — sub-Claudes implement against this)

```python
# === parallel_state ===
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1, 
    sequence_parallel: bool = False,
    context_parallel_size: int = 1,
) -> None: ...

def get_tensor_model_parallel_group() -> ProcessGroup: ...
def get_data_parallel_group() -> ProcessGroup: ...
def get_pipeline_model_parallel_group() -> ProcessGroup: ...
def get_tensor_model_parallel_rank() -> int: ...
def get_data_parallel_rank() -> int: ...
def get_pipeline_model_parallel_rank() -> int: ...
def get_tensor_model_parallel_world_size() -> int: ...
def get_data_parallel_world_size() -> int: ...

# === distributed ===
class DistributedDataParallel(MegatronModule):
    def __init__(self, config, ddp_config, module, ...): ...
    def forward(self, *args, **kwargs): ...
    def start_grad_sync(self): ...
    def finish_grad_sync(self): ...

def finalize_model_grads(model_chunks, num_tokens=None): ...

# === transformer ===
class TransformerConfig:
    hidden_size: int
    num_attention_heads: int
    num_layers: int
    # ... (dataclass)

class TransformerLayer(MegatronModule):
    def __init__(self, config, layer_number): ...
    def forward(self, hidden_states, attention_mask=None): ...

class TransformerBlock(MegatronModule):
    def __init__(self, config): ...
    def forward(self, hidden_states, attention_mask=None): ...

# === optimizer ===
class DistributedOptimizer:
    def __init__(self, optimizer, config, ...): ...
    def step(self): ...
    def zero_grad(self): ...
    def _compute_hetero_shard_boundaries(self, tier_map): ...

# === pipeline_parallel ===
def forward_backward_pipelining_without_interleaving(
    forward_step_func, data_iterator, model, 
    num_microbatches, dtype, ...
) -> List[dict]: ...

# === hetero_bridge ===
class HeteroBridge:
    @classmethod
    def install(cls, engine) -> None: ...
    
class TierMap:
    def __init__(self): ...  # auto-discovers GPU tiers
    def get_tier(self, rank: int) -> str: ...
    def get_compute_weight(self, rank: int) -> float: ...
```
