#!/bin/bash
# =========================================================================
# CLAUDE-23 SKILL: M318-M331 — EXACT repo/file/line references
# Author: dylanyunlong <dylanyunlong@gmail.com>
# =========================================================================
# RULES: net +400 lines/M#. Zero numpy.random. cat first, ast.parse after.
# INFRA at /home/claude/infra_repos/
# =========================================================================

# M318: deepspeed/runtime/utils.py (+400 net)
# DELETE lines 1485-1600. DIFF FROM:
#   Megatron-LM/megatron/core/optimizer/clip_grads.py:57        get_grad_norm_fp32()
#   Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:62  class Range
#   Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:118 _build_model_gbuf_param_range_map()
#   neuronx-distributed/src/neuronx_distributed/optimizer/zero_redundancy_optimizer.py:96 _clip_grad_norm()
#   TransformerEngine/examples/pytorch/comm_gemm_overlap/te_layer_with_overlap.py:175 _train()
#   Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:154 TensorParallelMuon

# M319: deepspeed/runtime/config.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:10 DistributedDataParallelConfig
#   Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:46 bucket_size
#   Megatron-LM/megatron/core/optimizer/optimizer_config.py:139 OptimizerConfig

# M320: deepspeed/runtime/constants.py (+400 net, delete 100 reserved). DIFF FROM:
#   nccl/src/device/all_reduce.h:234  RunWorkColl NCCL_ALGO_RING ALLREDUCE_CHUNKSTEPS
#   nccl/src/collectives.cc:111       ncclAllReduce() ncclInfo struct
#   Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:46 bucket_size

# M321: deepspeed/utils/comms_logging.py (+400 net). DIFF FROM:
#   nccl/plugins/profiler/example/nccl/profiler_v3.h  profiler event struct
#   nccl/plugins/profiler/example/event.h              event lifecycle
#   Megatron-LM/megatron/training/dgrad_logging.py     gradient logging
#   Megatron-LM/megatron/core/config_logger.py         log_config_to_disk()

# M322: deepspeed/utils/timer.py (+400 net). DIFF FROM:
#   flash-attention/flash_attn/utils/benchmark.py:8    benchmark_forward()
#   flash-attention/flash_attn/utils/benchmark.py:258  benchmark_memory()
#   TransformerEngine/benchmarks/attention/benchmark_dot_product_attention.py:44 timing
#   Megatron-LM/megatron/core/transformer/cuda_graphs.py is_graph_capturing()

# M323: deepspeed/comm/comm.py (+400 net). DIFF FROM:
#   veScale/vescale/dtensor/_collective_utils.py:66    mesh_scatter_ragged()
#   nccl/src/collectives.cc:111                        ncclAllReduce()
#   Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:~160 BucketGroup
#   neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:200 all_reduce()
#   neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:124 reduce_scatter()

# M324: deepspeed/comm/backend.py (+400 net). DIFF FROM:
#   TransformerEngine/transformer_engine/pytorch/ops/fused/userbuffers_forward_linear.py overlap
#   nccl/src/enqueue.cc                                ncclEnqueueCheck()
#   Megatron-LM/megatron/core/distributed/distributed_data_parallel.py overlap hooks

# M325: deepspeed/comm/torch.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py:9 WorkHandle
#   Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py:42 reduce_scatter()
#   neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:163 all_gather()
#   veScale/vescale/dtensor/_redistribute.py:130 class Redistribute

# M326: deepspeed/runtime/zero/stage_1_and_2.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:2696 step_with_ready_grads()
#   Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:118 _build_model_gbuf_param_range_map()
#   neuronx-distributed/src/neuronx_distributed/optimizer/zero_redundancy_optimizer.py:59 _shard_parameters()

# M327: deepspeed/ops/adam/fused_adam.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:154 TensorParallelMuon
#   Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:179 scaled_orthogonalize_fn
#   TransformerEngine/transformer_engine/common/multi_tensor/adam.cu fused kernel

# M328: deepspeed/runtime/bf16_optimizer.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/core/optimizer/grad_scaler.py:61 DynamicGradScaler
#   Megatron-LM/megatron/core/optimizer/optimizer.py BF16 master weight

# M329: deepspeed/runtime/lr_schedules.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/training/config/training_config.py LR fields
#   Megatron-LM/megatron/core/optimizer/optimizer.py LR in step()

# M330: deepspeed/runtime/pipe/engine.py (+400 net). DIFF FROM:
#   Megatron-LM/megatron/core/pipeline_parallel/schedules.py:592 forward_backward_no_pipelining()
#   Megatron-LM/megatron/core/pipeline_parallel/schedules.py:460 backward_step()
#   Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py combined schedule

# M331: REAL_GPU_BENCHMARK.py (+400 net, delete 80 M315 sim). DIFF FROM:
#   NKI-FA/exp_utils/draw_plot.py (da964f3)            parse_data() + seaborn
#   NKI-FA/hopper/benchmark_attn.py                    time_fwd() + flops()
#   flash-attention/flash_attn/utils/benchmark.py:8     benchmark_forward()
#   flash-attention/flash_attn/utils/benchmark.py:258   benchmark_memory()
