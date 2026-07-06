# Migration Status: `deepspeed/core/` vs `Megatron-LM/megatron/core/`

This document compares every subdirectory (and root-level files) between `deepspeed/core/` and `Megatron-LM/megatron/core/`, tracking which files have been migrated, which are unique additions in DeepSpeed, and which upstream Megatron-LM files have not yet been ported.

> **Legend**
>
> - ✅ **Full** — All upstream files present in DeepSpeed
> - 🟡 **Partial** — Some upstream files migrated; gaps remain
> - ❌ **Not started** — Upstream directory has no counterpart in DeepSpeed
> - 🆕 **DeepSpeed-only** — Directory or file exists only in `deepspeed/core/`

---

## Summary

| Subdirectory | Status | Shared | DS-only | MG-only | Notes |
|---|---|---|---|---|---|
| `datasets` | ✅ Full | 12 | 1 | 0 | All upstream files migrated; `__init__.py` added |
| `dist_checkpointing` | 🟡 Partial | 6 | 7 | 8 | Core mapping/strategies shared; both sides have unique files |
| `distributed` | 🟡 Partial | 4 | 1 | 11 | Core DDP migrated; FSDP subsystem not ported |
| `extensions` | ✅ Full | 3 | 1 | 0 | All upstream files migrated |
| `fusions` | 🟡 Partial | 2 | 1 | 3 | Base fusions migrated; MLA/routing fusions missing |
| `models` | 🟡 Partial | 2 | 2 | 28 | Only embeddings/rope_utils ported; most model defs missing |
| `optimizer` | 🟡 Partial | 6 | 0 | 6 | Core optimizer ported; advanced optimizers (Muon, etc.) missing |
| `pipeline_parallel` | 🟡 Partial | 4 | 2 | 3 | Core schedules ported; DS adds combined_1f1b |
| `quantization` | ✅ Full | 2 | 1 | 0 | All upstream files migrated |
| `resharding` | ✅ Full | 29 | 0 | 0 | Exact 1:1 match |
| `ssm` | ✅ Full | 16 | 0 | 0 | Exact 1:1 match |
| `tensor_parallel` | 🟡 Partial | 3 | 1 | 1 | DS keeps `mappings.py`; MG has `inference_layers.py` |
| `transformer` | 🟡 Partial | 18 | 2 | 15 | Core layers ported; CUDA graphs, experimental attention, inference dispatch missing |
| `hetero_bridge` | 🆕 DS-only | — | 8 | — | Heterogeneous hardware bridge (DeepSpeed addition) |
| `export` | ❌ Not started | — | — | 1 | TRT-LLM weight converter |
| `inference` | ❌ Not started | — | — | 70+ | Full inference engine, servers, sampling |
| `post_training` | ❌ Not started | — | — | 5 | ModelOpt integration |
| `tokenizers` | ❌ Not started | — | — | 23 | Text & vision tokenizers |
| Root files | 🟡 Partial | 22 | 2 | 1 | DS adds `desloc_config.py`, `stream_manager.py`; MG has `config_logger.py` |

---

## Per-directory detail

### `datasets` — ✅ Full

All 12 upstream Megatron-LM dataset files are present in DeepSpeed. DeepSpeed adds only a package `__init__.py`.

Shared files: `bert_dataset.py`, `blended_dataset.py`, `blended_megatron_dataset_builder.py`, `blended_megatron_dataset_config.py`, `commit_dataset.py`, `commitpack_streaming_dataset.py`, `data_schedule.py`, `gpt_dataset.py`, `indexed_dataset.py`, `masked_dataset.py`, `megatron_dataset.py`, `t5_dataset.py`.

---

### `dist_checkpointing` — 🟡 Partial

6 files are shared (core mapping, strategies for torch/zarr/two_stage, utils). Each side has significant unique content.

**DeepSpeed-only (7):** `__init__.py`, `core.py`, `dict_utils.py`, `optimizer.py`, `serialization.py`, `strategies/__init__.py`, `strategies/tensorstore.py`.

**Megatron-only (8):** `exchange_utils.py`, `state_dict_utils.py`, `strategies/async_utils.py`, `strategies/common.py`, `strategies/filesystem_async.py`, `strategies/nvrx.py`, `strategies/state_dict_saver.py`, `validation.py`.

Key gaps: async checkpointing strategies, NVRX strategy, and validation utilities have not been ported to DeepSpeed. DeepSpeed has its own TensorStore strategy and serialization layer.

---

### `distributed` — 🟡 Partial

Core DDP files are migrated (4 shared). The entire FSDP subsystem (11 files under `fsdp/`) and the standalone `distributed_data_parallel_config.py` remain Megatron-only.

**Megatron-only (11):** `distributed_data_parallel_config.py`, and 10 files under `fsdp/` including `mcore_fsdp_adapter.py`, `fully_shard.py`, `megatron_fsdp.py`, `mixed_precision.py`, `uneven_dtensor.py`, etc.

---

### `extensions` — ✅ Full

All 3 upstream files migrated: `kitchen.py`, `transformer_engine.py`, `transformer_engine_spec_provider.py`.

---

### `fusions` — 🟡 Partial

Base fusions (`fused_bias_geglu.py`, `fused_softmax.py`) are shared.

**Megatron-only (3):** `fused_indices_converter.py`, `fused_mla_yarn_rope_apply.py`, `fused_pad_routing_map.py`. These support MLA (Multi-Latent Attention) YaRN rope and MoE routing optimizations.

---

### `models` — 🟡 Partial

Only 2 files are shared (`__init__.py`, `common/embeddings/rope_utils.py`). This is the largest gap.

**Megatron-only (28):** Full model definitions for GPT, BERT, T5, Mamba, hybrid SSM, multimodal (LLaVA), vision (CLIP ViT, RADIO), MIMO, HuggingFace interop (FastConformer), and various layer specs / module specs. Also includes `backends.py`, `language_module.py`, `model_chunk_schedule_plan.py`, and YaRN rotary position embedding.

DeepSpeed has only the RoPE embedding utility; all concrete model architectures remain upstream.

---

### `optimizer` — 🟡 Partial

Core optimizer files are shared (6): `__init__.py`, `clip_grads.py`, `distrib_optimizer.py`, `optimizer.py`, `optimizer_config.py`, `param_layout.py`.

**Megatron-only (6):** `cpu_offloading/hybrid_optimizer.py`, `emerging_optimizers.py`, `layer_wise_optimizer.py`, `muon.py`, `optimizer_cuda_graph.py`, `qk_clip.py`. These cover CPU offloading, the Muon optimizer, CUDA graph support for optimizers, and QK clipping.

---

### `pipeline_parallel` — 🟡 Partial

Core scheduling and communication files are shared (4): `fine_grained_activation_offload.py`, `hybrid_cp_schedule.py`, `p2p_communication.py`, `schedules.py`.

**DeepSpeed-only (2):** `__init__.py`, `combined_1f1b.py` (a combined 1F1B schedule variant).

**Megatron-only (3):** `bridge_communicator.py`, `multimodule_communicator.py`, `utils.py`.

---

### `quantization` — ✅ Full

All upstream files migrated: `quant_config.py`, `utils.py`. DeepSpeed adds only `__init__.py`.

---

### `resharding` — ✅ Full

Exact 1:1 match across all 29 files, including the full NVSHMEM copy service subsystem.

---

### `ssm` — ✅ Full

Exact 1:1 match across all 16 files, covering Mamba layers, mixers, context parallelism, gated delta net, and all SSM ops (SSD, causal conv1d, etc.).

---

### `tensor_parallel` — 🟡 Partial

3 files shared: `__init__.py`, `layers.py`, `random.py`.

**DeepSpeed-only (1):** `mappings.py` — tensor-parallel communication mappings.

**Megatron-only (1):** `inference_layers.py` — inference-specific TP layers.

---

### `transformer` — 🟡 Partial

18 core files are shared, covering attention, MLP, MoE (experts, router, token dispatcher, fused A2A, shared experts, router replay, logging, utils), multi-latent attention, multi-token prediction, transformer block/config/layer, and module base.

**DeepSpeed-only (2):** `__init__.py`, `moe/__init__.py`.

**Megatron-only (15):** `cuda_graph_config.py`, `cuda_graphs.py`, `custom_layers/batch_invariant_kernels.py`, `enums.py`, `experimental_attention_variant/absorbed_mla.py`, `experimental_attention_variant/dsa.py`, `fsdp_dtensor_checkpoint.py`, `identity_op.py`, `moe/ops/__init__.py`, `moe/ops/paged_stash.py`, `moe/paged_stash.py`, `moe/token_dispatcher_inference.py`, `pipeline_parallel_layer_layout.py`, `spec_utils.py`, `utils.py`.

Key gaps: CUDA graph integration, experimental attention variants (absorbed MLA, DSA), FSDP DTensor checkpointing, inference token dispatch, and paged stash for MoE.

---

### `hetero_bridge` — 🆕 DeepSpeed-only

8 files. This is a DeepSpeed-specific addition for heterogeneous hardware support: `autosp_hook.py`, `desloc_sync_policy.py`, `dist_opt_adapter.py`, `engine_integration.py`, `pp_schedule_adapter.py`, `shard_planner.py`, `tier_map.py`, `__init__.py`.

---

### `export` — ❌ Not started

Megatron-LM has `trtllm/trtllm_weights_converter/single_device_trtllm_model_weights_converter.py` for TensorRT-LLM export. No counterpart in DeepSpeed.

---

### `inference` — ❌ Not started

Megatron-LM has 70+ files covering a full inference stack: dynamic/static engines, KV-cache contexts (block/chunk allocators), sampling (FlashInfer, torch), MoE inference (fused MoE, permute), quantization (MXFP8), text generation servers (OpenAI-compatible API), symmetric memory communication, scheduling, and model inference wrappers. None of this is present in DeepSpeed.

---

### `post_training` — ❌ Not started

Megatron-LM has 5 files for NVIDIA ModelOpt integration (GPT and hybrid model specs, custom layers). No counterpart in DeepSpeed.

---

### `tokenizers` — ❌ Not started

Megatron-LM has 23 files covering text tokenizers (HuggingFace, byte-level, null, Megatron-HF), vision tokenizers, chat templates, and reasoning/tool parsers (DeepSeek R1, Qwen3 Coder). No counterpart in DeepSpeed.

---

### Root files — 🟡 Partial

22 files are shared across both roots, covering core infrastructure: parallel state, model parallel config, FP4/FP8 utilities, CUDA graph support, communication grid, JIT, NCCL allocator, optimizer param scheduler, recompute, timers, enums, and more.

**DeepSpeed-only (2):** `desloc_config.py` (DeSLoC configuration), `stream_manager.py` (CUDA stream management).

**Megatron-only (1):** `config_logger.py`.

---

## Migration priorities

The largest functional gaps are:

1. **`models`** (28 files missing) — No concrete model architectures (GPT, BERT, T5, multimodal, vision) have been ported.
2. **`inference`** (70+ files missing) — The entire inference engine is absent.
3. **`transformer`** (15 files missing) — CUDA graphs, experimental attention, and inference-path components.
4. **`distributed`** (11 files missing) — The FSDP subsystem.
5. **`dist_checkpointing`** (8 files missing) — Async and NVRX checkpointing strategies.
6. **`optimizer`** (6 files missing) — Advanced optimizers and CUDA graph support.
7. **`tokenizers`** (23 files missing) — Full tokenizer stack.
8. **`post_training`** (5 files missing) — ModelOpt integration.
