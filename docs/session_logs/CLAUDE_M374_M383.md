Session: Claude-38 (M374-M383) | Base: commit f84d72d3

8 files changed, 77 insertions(+), 39 deletions(-)

| M# | File | What |
|---|---|---|
| M374 | deepspeed/compile/passes/sp_compile.py | AutoSPInputs NamedTuple return from prepare_autosp_inputs |
| M375 | deepspeed/compile/passes/sp_compile.py | A2A count uses torch.ops.autosp.all_to_all.default exact match |
| M376 | deepspeed/compile/passes/sp_compile.py | per-placeholder dtype reconciliation in pass_propagate_shapes |
| M377 | deepspeed/compile/init_sp.py | logger.warning for GQA fallback, store _effective_sp/dp_size |
| M378 | deepspeed/compile/passes/long_context_checkpointing.py | _ORIGINAL_SOLVE_MIN_CUT + restore_default_checkpointing + signature assertion |
| M379 | deepspeed/compile/custom_ops/sp_dp_registry.py | lower A2A drain threshold to HIGH_WATER//4, restore checkpointing on cleanup |
| M380 | deepspeed/compile/custom_ops/sp_compat.py | PyTorch upper-bound warning, CUDA arch >= 7.0 check |
| M381 | deepspeed/compile/custom_ops/all_to_all.py | ndim==4 assertion + contiguity guard on forward input |
| M382 | deepspeed/compile/fx.py | find_node_by_tag early-return, remove intermediate variable |
| M383 | deepspeed/compile/constants.py | add AUTOSP_ATTENTION_MASK_KEY for future mask sharding pass |

Infra repos cloned and grep-verified: Megatron-LM, NCCL, CCCL, TransformerEngine, PyTorch, BloomBee (6 repos)

Key patterns referenced:
- Megatron initialize_model_parallel (parallel_state.py:547) multi-dim mesh
- NCCL ncclGroupEndInternal (group.cc:753) pending ops drain
- PyTorch solve_min_cut (partitioners.py:2091) should_ban_recomputation
- CCCL DeviceHistogram::HistogramEven (device_histogram.cuh:49)
- TransformerEngine CommOverlap (extensions.h:674) A2A+GEMM pipeline
- BloomBee choose_best_blocks (block_selection.py:28) hetero peer selection

ZERO docstrings. ZERO comment lines. 65 net new executable lines.
