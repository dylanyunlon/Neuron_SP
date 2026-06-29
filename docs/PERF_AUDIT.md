# GPU Performance Commit Audit

Verifies whether GPU-tuning commits are on the live training path
(called by `desloc_engine.py` or `run_pretrain.py`).

## c37ea95f perf: 2497 → ~9 NCCL calls/step — bucketed sync + local SGD

- `deepspeed/runtime/zero3_hetero_shard.py` → engine:2 run:0
0 **LIVE**

## d89a62ba perf(distributed): fuse grad AllReduce for PCIe topology (M4149 pattern)

- `deepspeed/core/distributed/finalize_model_grads.py` → engine:8 run:0
0 **LIVE**

## 633ff4c4 feat(ddp): BF16 grad comm cast for PCIe bandwidth reduction (M3574)

- `deepspeed/core/distributed/distributed_data_parallel.py` → engine:0
0 run:0
0 **DEAD**
- `deepspeed/core/distributed/param_and_grad_buffer.py` → engine:1 run:0
0 **LIVE**

## 741d00d1 perf: delay grad norm all_reduce to end of accumulation window

- `deepspeed/runtime/desloc_engine.py` → engine:2 run:4 **LIVE**

## db3a925c perf: async sync_shard_to_model with CUDA stream overlap

- `deepspeed/runtime/desloc_engine.py` → engine:2 run:4 **LIVE**
- `deepspeed/runtime/zero3_hetero_shard.py` → engine:2 run:0
0 **LIVE**

