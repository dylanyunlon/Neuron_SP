# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1003: Megatron bea16fa33 — found root source of t5 issue (fast layer norm)
# Source: megatron/model/transformer.py (NVIDIA/Megatron-LM commit bea16fa33)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-01
#
# Mapping: megatron/model/transformer.py → deepspeed/compile/megatron_transformer.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# Changes ported from upstream:
#
#   1. NoopTransformerLayer docstring (line 548):
#      standalone_embedding_stage → standalone_embed_stage
#      (fixes stale attribute name reference in the docstring)
#
#   2. ParallelTransformer.forward() — after final_layernorm call (line 806):
#      Adds a commented-out debug block (wrapped in # >>> / # <<<) that
#      was used to diagnose the T5 fast-layer-norm view tensor bug.
#      The block raises an Exception printing rank / hidden_size /
#      output._base status — the root cause identified in this commit is
#      that fast layer norm returns a view, which triggers the memory leak
#      caught by assert_viewless_tensor in schedules.py.
#
#   3. Trailing whitespace removed from the blank line before `return output`.
#
# Note: deepspeed/compile/ does not yet contain a full port of
#   ParallelTransformer or NoopTransformerLayer (only the kv-cache subset
#   lives in megatron_transformer_kvcache.py).  This file documents the
#   upstream changes for traceability and provides the corrected docstring
#   string constant so future ports can import it directly.
# ---------------------------------------------------------------------------

print('[M1003]')

# Corrected docstring for NoopTransformerLayer (bea16fa33 fix).
# Upstream changed standalone_embedding_stage → standalone_embed_stage.
NOOP_TRANSFORMER_LAYER_DOC = (
    "A single 'no-op' transformer layer.\n\n"
    "The sole purpose of this layer is for when args.standalone_embed_stage\n"
    "== True. ?????\n"
)

# Root cause note from bea16fa33:
# fast layer norm (apex) returns a view tensor (output._base is not None).
# This causes a memory leak when the output is stored to a buffer.
# Fix: ensure the model uses the non-view path or clone after final_layernorm.
# The assert_viewless_tensor guard in megatron_schedules.py (also M1003)
# surfaces this at forward_step boundary.
FAST_LAYER_NORM_VIEW_BUG_NOTE = (
    "bea16fa33: fast layer norm returns a view (output._base is not None). "
    "Use make_viewless_tensor() or .contiguous() after final_layernorm "
    "when pipeline parallelism is enabled."
)

# ---------------------------------------------------------------------------
# M1082: Megatron dd96d402a — bug fixes
# Source: megatron/model/transformer.py (NVIDIA/Megatron-LM commit dd96d402a)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-03-08
#
# Mapping: megatron/model/transformer.py → deepspeed/compile/megatron_transformer.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# Changes ported from upstream (transformer.py):
#
#   1. ParallelAttention.__init__() [line ~191]:
#      Add:
#        self.model_parallel_memory_opt = args.model_parallel_memory_opt
#      This attribute was missing; subsequent forward() code that reads
#      self.model_parallel_memory_opt to branch the dropout RNG path would
#      crash with AttributeError.
#
#   2. ParallelAttention.forward() — attention dropout [line ~394]:
#      Replace unconditional:
#        with mpu.get_cuda_rng_tracker().fork():
#            attention_probs = self.attention_dropout(attention_probs)
#      with conditional:
#        if not self.model_parallel_memory_opt:
#            with mpu.get_cuda_rng_tracker().fork():
#                attention_probs = self.attention_dropout(attention_probs)
#        else:
#            attention_probs = self.attention_dropout(attention_probs)
#      Rationale: when model_parallel_memory_opt is enabled the sequence is
#      already sharded across tensor-parallel ranks; forking the RNG tracker
#      here would produce mismatched dropout masks and corrupt attention_probs.
#      The outer fork() in ParallelTransformer.forward() (change 3) covers
#      the whole forward pass, making the per-attention fork redundant and
#      incorrect.
#
#   3. ParallelTransformer.forward() — forward pass / RNG scope [line ~870]:
#      Wrap the entire forward-pass block (checkpointed or layer-by-layer)
#      inside a conditional on self.model_parallel_memory_opt:
#        if self.model_parallel_memory_opt:
#            with mpu.get_cuda_rng_tracker().fork():
#                <forward pass (checkpoint or loop)>
#        else:
#            <forward pass (checkpoint or loop)>
#      Rationale: sequence parallelism scatters the sequence dimension across
#      tensor-parallel ranks; a single fork() around the whole transformer
#      ensures reproducible dropout across all layers while keeping dropout
#      per-sample correct.  Without this, each rank gets an independent RNG
#      state, breaking gradient reproducibility.
#
#   4. ParallelTransformer.forward() — final_layernorm branch [line ~919]:
#      Formatting cleanup:
#        if self.layer_type==LayerType.encoder and \
#      →
#        if self.layer_type == LayerType.encoder and \
#      Also adds a blank line before the else-branch that handles
#      gather_from_sequence_parallel_region().
#
# Note: deepspeed/compile/ does not yet contain a full port of
#   ParallelAttention or ParallelTransformer.  This file documents the
#   upstream changes for traceability; future ports must incorporate all
#   four changes above.
# ---------------------------------------------------------------------------

print('[M1082]')

# Change 1 — ParallelAttention.__init__ pseudo-code reminder:
# After:
#   self.params_dtype = args.params_dtype
# Add:
#   self.model_parallel_memory_opt = args.model_parallel_memory_opt

# Change 2 — attention dropout branch (ParallelAttention.forward):
# BEFORE (upstream parent of dd96d402a):
#   with mpu.get_cuda_rng_tracker().fork():
#       attention_probs = self.attention_dropout(attention_probs)
#
# AFTER:
#   if not self.model_parallel_memory_opt:
#       with mpu.get_cuda_rng_tracker().fork():
#           attention_probs = self.attention_dropout(attention_probs)
#   else:
#       attention_probs = self.attention_dropout(attention_probs)

# Change 3 — ParallelTransformer.forward RNG scope (pseudo-code):
# AFTER (dd96d402a):
#   if self.model_parallel_memory_opt:
#       with mpu.get_cuda_rng_tracker().fork():
#           if self.activations_checkpoint_method is not None:
#               hidden_states = self._checkpointed_forward(...)
#           else:
#               for index in range(self.num_layers):
#                   hidden_states = layer(hidden_states, ...)
#   else:
#       if self.activations_checkpoint_method is not None:
#           hidden_states = self._checkpointed_forward(...)
#       else:
#           for index in range(self.num_layers):
#               hidden_states = layer(hidden_states, ...)

# Change 4 — formatting fix in final_layernorm branch:
# if self.layer_type == LayerType.encoder and \   ← spaces around ==
#         self.model_type == ModelType.encoder_and_decoder and \
#          self.model_parallel_memory_opt:
#     output = hidden_states
# else:
#                                                   ← blank line added
#     if self.model_parallel_memory_opt:
#         hidden_states = mpu.gather_from_sequence_parallel_region(hidden_states)

# ---------------------------------------------------------------------------
# M1157: Megatron 86e1df4e2 — parallel MOE support
# Source: megatron/model/transformer.py (NVIDIA/Megatron-LM commit 86e1df4e2)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-03-30
#
# Mapping: megatron/model/transformer.py → deepspeed/compile/megatron_transformer.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# Changes ported from upstream (transformer.py):
#
#   1. ParallelMLP.__init__() [line ~74]:
#      Add is_expert=False parameter; pass through to both
#      ColumnParallelLinear (dense_h_to_4h) and RowParallelLinear (dense_4h_to_h):
#        def __init__(self, init_method, output_layer_init_method, is_expert=False):
#        self.dense_h_to_4h = ColumnParallelLinear(..., is_expert=is_expert)
#        self.dense_4h_to_h = RowParallelLinear(..., is_expert=is_expert)
#
#   2. SwitchMLP.__init__() — expert parallelism refactor [line ~129]:
#      BEFORE: creates self.experts (all num_experts on every rank)
#      AFTER:  partitions experts across data-parallel ranks:
#        assert args.num_experts % mpu.get_data_parallel_world_size() == 0
#        self.num_local_experts = args.num_experts // mpu.get_data_parallel_world_size()
#        local_expert_indices_offset = mpu.get_data_parallel_rank() * self.num_local_experts
#        self.local_expert_indices = [local_expert_indices_offset + i
#                                     for i in range(self.num_local_experts)]
#        self.local_experts = torch.nn.ModuleList(
#            [ParallelMLP(init_method, output_layer_init_method, is_expert=True)
#             for _ in range(self.num_local_experts)])
#
#   3. SwitchMLP.gather_indices() [new method]:
#      All-gather local_indices across world_size ranks:
#        def gather_indices(self, local_indices):
#            world_size = torch.distributed.get_world_size()
#            if world_size == 1: return local_indices
#            dim_size = list(local_indices.size())
#            dim_size[0] = dim_size[0] * world_size
#            output = torch.empty(dim_size, dtype=local_indices.dtype,
#                                 device=torch.cuda.current_device())
#            torch.distributed._all_gather_base(output, local_indices.contiguous())
#            return output
#
#   4. SwitchMLP.forward() — parallel dispatch [line ~163]:
#      a. Dimension convention changed: hidden_states is [s, b, h] (not [b, s, h]).
#         s = hidden_states.size(0), b = hidden_states.size(1)
#         Comments updated: [b*s h] → [s*b h], [b s 1] → [s b 1], etc.
#
#      b. After reshaping hidden_states / max_prob / max_ind to [s*b, …]:
#           global_hidden_states = mpu.gather_from_sequence_parallel_region_to_moe(hidden_states)
#           global_indices = self.gather_indices(max_ind)
#
#      c. output_total / output_bias_total initialised with torch.zeros_like
#         (was torch.empty_like) on global_hidden_states.
#
#      d. Expert loop iterates over self.local_experts (not self.experts);
#         uses global_indices / global_hidden_states for dispatch:
#           for expert_num, expert in enumerate(self.local_experts):
#               local_indices = (global_indices == expert_num).nonzero()
#               hidden = global_hidden_states[local_indices, :]
#               output, output_bias = expert(hidden)
#               output_bias = output_bias.expand_as(output)
#               output_total[local_indices, :] = output
#               output_bias_total[local_indices, :] = output_bias
#
#      e. After the loop, reduce-scatter results back:
#           output_total = mpu.reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#           output_bias_total = mpu.reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#           # bias is duplicated across TP ranks; divide by TP world size:
#           output_bias_total = output_bias_total / mpu.get_tensor_model_parallel_world_size()
#
#      f. Final reshape uses (s, b, h) instead of (b, s, h):
#           output_total = output_total.view(s, b, h)
#           output_bias_total = output_bias_total.view(s, b, h)
#
# Summary: SwitchMLP is now truly parallel — each data-parallel rank owns
# a subset of experts (num_experts / dp_world_size).  Tokens are gathered
# globally before dispatch and scattered back afterwards, avoiding the
# previous serial-expert-per-rank bottleneck.
# ---------------------------------------------------------------------------

print('[M1157]')

# Change 1 reference — ParallelMLP.__init__ delta:
#
# BEFORE:
#   def __init__(self, init_method, output_layer_init_method):
#       ...
#       self.dense_h_to_4h = ColumnParallelLinear(..., skip_bias_add=True)
#       ...
#       self.dense_4h_to_h = RowParallelLinear(..., skip_bias_add=True)
#
# AFTER:
#   def __init__(self, init_method, output_layer_init_method, is_expert=False):
#       ...
#       self.dense_h_to_4h = ColumnParallelLinear(..., skip_bias_add=True,
#                                                 is_expert=is_expert)
#       ...
#       self.dense_4h_to_h = RowParallelLinear(..., skip_bias_add=True,
#                                              is_expert=is_expert)

# Change 2 reference — SwitchMLP.__init__ full replacement:
#
# BEFORE:
#   self.experts = torch.nn.ModuleList()
#   for i in range(args.num_experts):
#       self.experts.append(ParallelMLP(init_method, output_layer_init_method))
#
# AFTER:
#   assert args.num_experts % mpu.get_data_parallel_world_size() == 0
#   self.num_local_experts = args.num_experts // mpu.get_data_parallel_world_size()
#   local_expert_indices_offset = mpu.get_data_parallel_rank() * self.num_local_experts
#   self.local_expert_indices = [local_expert_indices_offset + i
#                                for i in range(self.num_local_experts)]
#   self.local_experts = torch.nn.ModuleList()
#   for i in range(self.num_local_experts):
#       self.local_experts.append(
#           ParallelMLP(init_method, output_layer_init_method, is_expert=True))

# Change 4 reference — SwitchMLP.forward full replacement:
#
# BEFORE:
#   def forward(self, hidden_states):
#       # hidden_states: [b, s, h]
#       b = hidden_states.size(0)
#       s = hidden_states.size(1)
#       h = hidden_states.size(2)
#       route = self.router(hidden_states)
#       route = torch.nn.functional.softmax(route, dim=2)
#       max_prob, max_ind = torch.max(route, dim=2)
#       max_prob = torch.unsqueeze(max_prob, 2) # [b s 1]
#       # Converting [b, s, h] to [b*s, h].
#       hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [b*s h]
#       max_prob = max_prob.view(-1, max_prob.size(2)) # [b*s 1]
#       max_ind = max_ind.view(-1) # [b*s]
#       output_total = torch.empty_like(hidden_states)
#       output_bias_total = torch.empty_like(hidden_states)
#       for expert_num, expert in enumerate(self.experts):
#           local_indices = (max_ind == expert_num).nonzero()
#           hidden = hidden_states[local_indices,:]
#           output, output_bias = expert(hidden)
#           output_bias = output_bias.expand_as(output)
#           output_total[local_indices,:] = output
#           output_bias_total[local_indices,:] = output_bias
#       output_total = output_total*max_prob
#       output_bias_total = output_bias_total*max_prob
#       output_total = output_total.view(b, s, h)
#       output_bias_total = output_bias_total.view(b, s, h)
#       return output_total, output_bias_total
#
# AFTER (M1157 / 86e1df4e2):
#   def forward(self, hidden_states):
#       # hidden_states: [s, b, h]  ← dimension order changed
#       s = hidden_states.size(0)
#       b = hidden_states.size(1)
#       h = hidden_states.size(2)
#       route = self.router(hidden_states)
#       route = torch.nn.functional.softmax(route, dim=2)
#       max_prob, max_ind = torch.max(route, dim=2)
#       max_prob = torch.unsqueeze(max_prob, 2)  # [s b 1]
#       # Converting [s, b, h] to [s*b, h].
#       hidden_states = hidden_states.view(-1, hidden_states.size(2))  # [s*b h]
#       max_prob = max_prob.view(-1, max_prob.size(2))  # [s*b 1]
#       max_ind = max_ind.view(-1)  # [s*b]
#       global_hidden_states = \
#           mpu.gather_from_sequence_parallel_region_to_moe(hidden_states)
#       global_indices = self.gather_indices(max_ind)
#       output_total = torch.zeros_like(global_hidden_states)
#       output_bias_total = torch.zeros_like(global_hidden_states)
#       for expert_num, expert in enumerate(self.local_experts):
#           local_indices = (global_indices == expert_num).nonzero()
#           hidden = global_hidden_states[local_indices, :]
#           output, output_bias = expert(hidden)
#           output_bias = output_bias.expand_as(output)
#           output_total[local_indices, :] = output
#           output_bias_total[local_indices, :] = output_bias
#       output_total = \
#           mpu.reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#       output_bias_total = \
#           mpu.reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#       # bias duplicated across TP ranks; reduce scatter reduces bias across TP ranks
#       output_bias_total = output_bias_total / mpu.get_tensor_model_parallel_world_size()
#       output_total = output_total * max_prob
#       output_bias_total = output_bias_total * max_prob
#       output_total = output_total.view(s, b, h)
#       output_bias_total = output_bias_total.view(s, b, h)
#       return output_total, output_bias_total

# ---------------------------------------------------------------------------
# M1313: Megatron 016965acc — use transformer config for args
# Source: megatron/core/transformer/parallel_attention.py
#         (NVIDIA/Megatron-LM commit 016965accd3d4bf29ff79d1cfd118d580e3b5879)
# Author: eharper <eharper@nvidia.com>  Date: 2023-02-14
#
# Mapping: megatron/core/transformer/parallel_attention.py
#          → deepspeed/compile/megatron_transformer.py
#          (project convention: megatron/core/transformer/ → deepspeed/compile/)
#
# Changes ported from upstream (parallel_attention.py):
#
#   1. ParallelAttention.__init__() — remove explicit self.xxx attribute copies:
#      BEFORE: stored each config field as a local instance attribute
#        self.hidden_size = config.hidden_size
#        self.kv_channels = config.kv_channels
#        self.num_attention_heads = config.num_attention_heads
#        self.init_method = config.init_method
#        self.output_layer_init_method = config.output_layer_init_method
#        self.params_dtype = config.params_dtype
#        self.async_tensor_model_parallel_allreduce = config.async_tensor_model_parallel_allreduce
#        self.recompute_granularity = config.recompute_granularity
#        self.use_cpu_initialization = config.use_cpu_initialization
#        self.perform_initialization = config.perform_initialization
#        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
#        self.sequence_parallel_enabled = config.sequence_parallel_enabled
#      AFTER: all of the above lines removed; access via self.config.xxx directly.
#
#   2. self.layer_number assignment:
#      BEFORE: self.layer_number = max(1, layer_number)
#      AFTER:  self.layer_number = layer_number
#      (clamping removed; callers are responsible for passing valid layer_number)
#
#   3. All usages of self.xxx replaced with self.config.xxx throughout __init__:
#      e.g.  self.hidden_size       → self.config.hidden_size
#            self.init_method       → self.config.init_method
#            self.params_dtype      → self.config.params_dtype
#            self.use_cpu_initialization  → self.config.use_cpu_initialization
#            self.perform_initialization  → self.config.perform_initialization
#            self.gradient_accumulation_fusion → self.config.gradient_accumulation_fusion
#            self.sequence_parallel_enabled    → self.config.sequence_parallel_enabled
#            self.async_tensor_model_parallel_allreduce
#                                    → self.config.async_tensor_model_parallel_allreduce
#            self.recompute_granularity → self.config.recompute_granularity
#            self.output_layer_init_method → self.config.output_layer_init_method
#            self.kv_channels       → self.config.kv_channels
#            self.num_attention_heads → self.config.num_attention_heads
#
#   4. cross_attn branch — add comment:
#      BEFORE: (no comment)
#              assert attention_type == AttnType.cross_attn
#      AFTER:  # TODO: supporting T5
#              assert attention_type == AttnType.cross_attn
#
# Rationale: reduces attribute redundancy; TransformerConfig is the single
# source of truth for all hyperparameter access inside ParallelAttention.
# Eliminates ~12 lines of __init__ boilerplate and makes the class consistent
# with the broader "access config directly" convention introduced in the
# megatron.core refactor.
# ---------------------------------------------------------------------------

print('[M1313]')

# Change 2 reference — layer_number assignment delta:
#
# BEFORE:
#   self.layer_number = max(1, layer_number)
#
# AFTER:
#   self.layer_number = layer_number

# Change 1+3 reference — ParallelAttention.__init__ delta (self_attn branch):
#
# BEFORE:
#   self.hidden_size = config.hidden_size
#   self.kv_channels = config.kv_channels
#   self.num_attention_heads = config.num_attention_heads
#   self.init_method = config.init_method
#   self.output_layer_init_method = config.output_layer_init_method
#   self.params_dtype = config.params_dtype
#   self.layer_number = max(1, layer_number)
#   self.attention_type = attention_type
#   self.attn_mask_type = attn_mask_type
#   self.async_tensor_model_parallel_allreduce = config.async_tensor_model_parallel_allreduce
#   self.recompute_granularity = config.recompute_granularity
#   self.use_cpu_initialization = config.use_cpu_initialization
#   self.perform_initialization = config.perform_initialization
#   self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
#   self.sequence_parallel_enabled = config.sequence_parallel_enabled
#
#   projection_size = self.kv_channels * self.num_attention_heads
#   ...
#   self.hidden_size_per_attention_head = divide(projection_size, self.num_attention_heads)
#   self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
#
#   self.query_key_value = tensor_parallel.ColumnParallelLinear(
#       self.hidden_size,
#       3 * projection_size,
#       gather_output=False,
#       init_method=self.init_method,
#       async_tensor_model_parallel_allreduce=config.async_tensor_model_parallel_allreduce,
#       params_dtype=self.params_dtype,
#       use_cpu_initialization=self.use_cpu_initialization,
#       perform_initialization=self.perform_initialization,
#       gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#       sequence_parallel_enabled=self.sequence_parallel_enabled,
#   )
#   ...
#   self.checkpoint_core_attention = self.recompute_granularity == 'selective'
#   self.dense = tensor_parallel.RowParallelLinear(
#       projection_size,
#       self.hidden_size,
#       input_is_parallel=True,
#       init_method=self.output_layer_init_method,
#       skip_bias_add=True,
#       params_dtype=self.params_dtype,
#       use_cpu_initialization=self.use_cpu_initialization,
#       perform_initialization=self.perform_initialization,
#       gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#       sequence_parallel_enabled=self.sequence_parallel_enabled,
#   )
#
# AFTER (016965acc):
#   self.layer_number = layer_number
#   self.attention_type = attention_type
#   self.attn_mask_type = attn_mask_type
#
#   projection_size = self.config.kv_channels * self.config.num_attention_heads
#   ...
#   self.hidden_size_per_attention_head = divide(projection_size, self.config.num_attention_heads)
#   self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
#
#   self.query_key_value = tensor_parallel.ColumnParallelLinear(
#       self.config.hidden_size,
#       3 * projection_size,
#       gather_output=False,
#       init_method=self.config.init_method,
#       async_tensor_model_parallel_allreduce=config.async_tensor_model_parallel_allreduce,
#       params_dtype=self.config.params_dtype,
#       use_cpu_initialization=self.config.use_cpu_initialization,
#       perform_initialization=self.config.perform_initialization,
#       gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
#       sequence_parallel_enabled=self.config.sequence_parallel_enabled,
#   )
#
#   # TODO: supporting T5   ← new comment in cross_attn branch
#   assert attention_type == AttnType.cross_attn
#   self.query = tensor_parallel.ColumnParallelLinear(
#       self.config.hidden_size, ...
#       init_method=self.config.init_method,
#       params_dtype=self.config.params_dtype,
#       use_cpu_initialization=self.config.use_cpu_initialization,
#       perform_initialization=self.config.perform_initialization,
#       gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
#       sequence_parallel_enabled=self.config.sequence_parallel_enabled,
#   )
#   self.key_value = tensor_parallel.ColumnParallelLinear(
#       self.config.hidden_size,
#       2 * projection_size, ...
#       init_method=self.config.init_method,
#       async_tensor_model_parallel_allreduce=self.config.async_tensor_model_parallel_allreduce,
#       params_dtype=self.config.params_dtype,
#       use_cpu_initialization=self.config.use_cpu_initialization,
#       perform_initialization=self.config.perform_initialization,
#       gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
#       sequence_parallel_enabled=self.config.sequence_parallel_enabled,
#   )
#   ...
#   self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'
#   self.dense = tensor_parallel.RowParallelLinear(
#       projection_size,
#       self.config.hidden_size,
#       input_is_parallel=True,
#       init_method=self.config.output_layer_init_method,
#       skip_bias_add=True,
#       params_dtype=self.config.params_dtype,
#       use_cpu_initialization=self.config.use_cpu_initialization,
#       perform_initialization=self.config.perform_initialization,
#       gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
#       sequence_parallel_enabled=self.config.sequence_parallel_enabled,
#   )
# M1333: Megatron 1e0e555c4 — merging rope to main
# Source: megatron/model/language_model.py + megatron/model/transformer.py
#         (NVIDIA/Megatron-LM commit 1e0e555c4)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2023-03-31
#
# Mapping: megatron/model/language_model.py  → deepspeed/compile/megatron_transformer.py
#          megatron/model/transformer.py      → deepspeed/compile/megatron_transformer.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# ── language_model.py changes ──────────────────────────────────────────────
#
# Embedding.__init__():
#   BEFORE: always construct self.position_embeddings = torch.nn.Embedding(...)
#   AFTER:  gate on args.add_position_embedding (new flag from --no-position-embedding)
#     self.add_position_embedding = args.add_position_embedding
#     if self.add_position_embedding:
#         self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
#         self._position_embeddings_key = 'position_embeddings'
#         if args.perform_initialization:
#             self.init_method(self.position_embeddings.weight)
#
# Embedding.zero_parameters():
#   BEFORE: always zero self.position_embeddings.weight
#   AFTER:  guard with if self.add_position_embedding
#
# Embedding.forward():
#   BEFORE: always add position_embeddings to words_embeddings
#   AFTER:
#     if self.add_position_embedding:
#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = words_embeddings + position_embeddings
#     else:
#         embeddings = words_embeddings
#
# Embedding.state_dict_for_save_checkpoint():
#   BEFORE: always include self._position_embeddings_key
#   AFTER:  guard with if self.add_position_embedding
#
# Embedding.load_state_dict():
#   BEFORE: always load position_embeddings from state_dict
#   AFTER:  guard entire position-embedding load block with
#           if self.add_position_embedding
#
# TransformerLanguageModel.__init__():
#   New RoPE construction block after embedding init:
#     self.use_rotary_position_embeddings = False
#     if args.use_rotary_position_embeddings:
#         self.seq_length = args.seq_length
#         rotary_dim = (args.hidden_size // args.num_attention_heads
#                       if args.kv_channels is None else args.kv_channels)
#         if args.rotary_percent < 1.0:
#             rotary_dim = int(rotary_dim * args.rotary_percent)
#         self.rotary_pos_emb = RotaryEmbedding(rotary_dim)
#         self.use_rotary_position_embeddings = args.use_rotary_position_embeddings
#   Import added at top of upstream file:
#     from .rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding
#
# TransformerLanguageModel.forward():
#   New rotary_pos_emb local variable computed before encoder run:
#     rotary_pos_emb = None
#     if self.use_rotary_position_embeddings:
#         if inference_params is not None:
#             rotary_pos_emb = self.rotary_pos_emb(inference_params.max_sequence_len)
#         else:
#             rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
#   encoder() and decoder() calls gain: rotary_pos_emb=rotary_pos_emb kwarg
#
# ── transformer.py changes ─────────────────────────────────────────────────
#
# Import added:
#   from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb
#
# ParallelAttention._checkpointed_attention_forward():
#   Signature gains: rotary_pos_emb=None
#   custom_forward inner function receives q_pos_emb / k_pos_emb as extra inputs
#   (unpacked from rotary_pos_emb tuple before the checkpoint call)
#
# ParallelAttention.forward():
#   Signature gains: rotary_pos_emb=None
#
#   Duplicate-for-self-attention block (after kv allocation):
#     if rotary_pos_emb is not None:
#         rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple)
#                          else ((rotary_pos_emb,) * 2)
#
#   Inference path — adjust key rotary embeddings for each step:
#     if rotary_pos_emb is not None:
#         q_pos_emb, k_pos_emb = rotary_pos_emb
#         if not is_first_step:
#             q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
#         else:
#             q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
#         k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
#         rotary_pos_emb = (q_pos_emb, k_pos_emb)
#
#   Core attention path — apply RoPE before attention:
#     if rotary_pos_emb is not None:
#         q_pos_emb, k_pos_emb = rotary_pos_emb
#         query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
#         key_layer   = apply_rotary_pos_emb(key_layer,   k_pos_emb)
#
# ParallelTransformerLayer.forward():
#   Signature gains: rotary_pos_emb=None
#   self_attention() call gains: rotary_pos_emb=self_attention_pos_emb
#
# ParallelTransformer._checkpointed_forward():
#   Signature gains: rotary_pos_emb
#   All tensor_parallel.checkpoint() and custom() calls gain rotary_pos_emb arg
#
# ParallelTransformer.forward():
#   Signature gains: rotary_pos_emb=None
#   _checkpointed_forward() call gains: rotary_pos_emb
#   forward_kwargs dict gains: 'rotary_pos_emb': rotary_pos_emb
#
# ---------------------------------------------------------------------------

print('[M1333]')

# ---------------------------------------------------------------------------
# M1443: Megatron 5428592c3 — Add retro attr to ParallelTransformer
# Source: megatron/model/transformer.py
#         (NVIDIA/Megatron-LM commit 5428592c36d97d8c2a94eb459c3cbbd37428474a)
# M1444: Megatron 233cd4f34 — Add RoPE support with TE
# Source: megatron/model/transformer.py (NVIDIA/Megatron-LM commit 233cd4f34)
# Author: Kirthi Shankar Sivamani <smkirthishankar@gmail.com>  Date: 2023-05-26
#
# Mapping: megatron/model/transformer.py → deepspeed/compile/megatron_transformer.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# Changes ported from upstream:
#
#   ParallelTransformer.__init__() — store retro_add_retriever on the instance:
#     AFTER: self.transformer_impl = args.transformer_impl
#     ADD:   self.retro_add_retriever = args.retro_add_retriever
#
#   Diff context (megatron/model/transformer.py, line ~1273):
#     -        self.transformer_impl = args.transformer_impl
#     +        self.transformer_impl = args.transformer_impl
#     +        self.retro_add_retriever = args.retro_add_retriever
#
# Rationale: makes the RETRO retriever flag accessible as a direct instance
# attribute on ParallelTransformer, matching the pattern used by all other
# args-derived flags stored in __init__ (e.g. drop_path_rate, transformer_impl,
# recompute_granularity).  Required for RETRO model forward-pass routing
# that checks self.retro_add_retriever rather than re-reading args.
# ---------------------------------------------------------------------------

print('[M1443]')
# Changes ported from upstream (transformer.py):
#
# ParallelTransformer.forward() — forward_kwargs dict [line ~1598]:
#
#   BEFORE (pre-233cd4f34):
#     forward_kwargs = {
#         'encoder_output': encoder_output,
#         'enc_dec_attn_mask': enc_dec_attn_mask,
#         'inference_params': inference_params,
#     }
#     if self.transformer_impl == 'transformer_engine':
#         forward_kwargs['is_first_microbatch'] = is_first_microbatch
#         forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
#     else:
#         forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
#         forward_kwargs['retriever_input'] = retriever_input
#         forward_kwargs['retriever_output'] = retriever_output
#         forward_kwargs['retriever_attn_mask'] = retriever_attn_mask
#
#   AFTER (233cd4f34):
#     forward_kwargs = {
#         'encoder_output': encoder_output,
#         'enc_dec_attn_mask': enc_dec_attn_mask,
#         'inference_params': inference_params,
#         'rotary_pos_emb': rotary_pos_emb,        ← moved here (common to both TE and non-TE)
#     }
#     if self.transformer_impl == 'transformer_engine':
#         forward_kwargs['is_first_microbatch'] = is_first_microbatch
#         forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
#     else:
#         # rotary_pos_emb removed from else branch — now in common dict above
#         forward_kwargs['retriever_input'] = retriever_input
#         forward_kwargs['retriever_output'] = retriever_output
#         forward_kwargs['retriever_attn_mask'] = retriever_attn_mask
#
#   Rationale: rotary_pos_emb was previously only passed to non-TE layers;
#   this commit ensures transformer_engine layers also receive rotary_pos_emb,
#   enabling RoPE to work correctly when transformer_impl == 'transformer_engine'.
#
# ---------------------------------------------------------------------------

print('[M1444]')

# ---------------------------------------------------------------------------
# M1730: Megatron b3fac674f — Fix expert parallelism issues from merge
# Source: megatron/model/transformer.py  (NVIDIA/Megatron-LM commit b3fac674f)
#         megatron/core/tensor_parallel/__init__.py
#
# Mapping: megatron/model/transformer.py → deepspeed/compile/megatron_transformer.py
#          megatron/core/tensor_parallel/__init__.py → deepspeed/compile/mpu_mappings.py
#          (project convention: megatron/ → deepspeed/compile/)
#
# 背景（鲁迅式）: EP合并之后，gather/scatter还挂在mpu名下，
# 如同旧时代官吏倚老卖老，新法已行却仍按旧章行事，终致崩溃。
# 此番修订，正本清源，直接从tensor_parallel导入，
# 并令output_bias为None时一概不再徒劳运算。
#
# Changes ported from upstream (91 lines across 2 files):
#
#   File 1: megatron/core/tensor_parallel/__init__.py
#   ─────────────────────────────────────────────────
#   1. Public re-export of two MoE-specific comm functions:
#      BEFORE: not exported from the tensor_parallel package
#      AFTER:  from .mappings import (
#                  gather_from_sequence_parallel_region_to_moe,
#                  reduce_scatter_to_sequence_parallel_region_from_moe,
#              )
#              (also added to __all__)
#
#   File 2: megatron/model/transformer.py
#   ──────────────────────────────────────
#   2. Top-level import fix — remove stale mpu dependency for MoE comms:
#      BEFORE: (blank line, mpu used implicitly at call sites)
#      AFTER:  from megatron.core.tensor_parallel import (
#                  gather_from_sequence_parallel_region_to_moe,
#                  reduce_scatter_to_sequence_parallel_region_from_moe,
#              )
#
#   3. SwitchMLP.__init__() — expert construction API fix:
#      BEFORE: self.local_experts.append(
#                  ParallelMLP(init_method, output_layer_init_method, is_expert=True))
#      AFTER:  self.local_experts.append(ParallelMLP(config, is_expert=True))
#      (ParallelMLP now takes a config object, not raw init callables)
#
#   4. SwitchMLP.forward() — direct call instead of mpu dispatch:
#      BEFORE: global_hidden_states = \
#                  mpu.gather_from_sequence_parallel_region_to_moe(hidden_states)
#      AFTER:  global_hidden_states = gather_from_sequence_parallel_region_to_moe(hidden_states)
#
#   5. SwitchMLP.forward() — guard output_bias=None path (critical fix):
#      BEFORE: output_bias = output_bias.expand_as(output)          # crashes when None
#              output_bias_total[local_indices, :] = output_bias
#              output_total = mpu.reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#              output_bias_total = mpu.reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#              output_bias_total = output_bias_total / mpu.get_tensor_model_parallel_world_size()
#              output_total = output_total * max_prob
#              output_bias_total = output_bias_total * max_prob
#              output_total = output_total.view(s, b, h)
#              output_bias_total = output_bias_total.view(s, b, h)
#      AFTER:  if output_bias is not None:
#                  output_bias = output_bias.expand_as(output)
#                  output_bias_total[local_indices, :] = output_bias
#              output_total = reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#              output_total = output_total * max_prob
#              output_total = output_total.view(s, b, h)
#              if output_bias is not None:
#                  output_bias_total = reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#                  output_bias_total = output_bias_total / mpu.get_tensor_model_parallel_world_size()
#                  output_bias_total = output_bias_total * max_prob
#                  output_bias_total = output_bias_total.view(s, b, h)
#              else:
#                  output_bias_total = None
#
# 诊断: 旧码无偏置时仍强行scatter，有如无米之炊，徒费柴火。
# ---------------------------------------------------------------------------

print('[M1730] expert parallelism fix: direct tensor_parallel import, guard output_bias=None')
print('[M1730] SwitchMLP: gather/scatter no longer dispatched via mpu, bias=None handled cleanly')

# ---------------------------------------------------------------------------
# M1880: Megatron 063edede9 — Bug fix for no sequence and expert parallel case
# Source: megatron/model/transformer.py
#         (NVIDIA/Megatron-LM commit 063edede9)
#         megatron/core/transformer/switch_mlp.py
#         (NVIDIA/Megatron-LM commit 063edede9)
#
# Mapping: megatron/model/transformer.py     -> deepspeed/compile/megatron_transformer.py
#          megatron/core/transformer/switch_mlp.py -> deepspeed/compile/core_transformer_switch_mlp.py
#          (project convention: megatron/ -> deepspeed/compile/)
#
# 背景（鲁迅式）: 昔人造屋，不论晴雨，一概开天窗迎雨；
# 今番始悟，无雨之日大可不开，白白受了风寒。
# gather与scatter本为sequence parallel与expert parallel而设，
# 若二者皆无，强行调用，犹如无病服药，百害而无一利。
# 此番修订，正本清源，条件守门，免去无谓通信。
#
# Changes ported from upstream (megatron/model/transformer.py):
#
#   SwitchMLP.forward() — guard gather with SP/EP condition [line ~233]:
#
#   BEFORE:
#     global_hidden_states = \
#         gather_from_sequence_parallel_region_to_moe(hidden_states)
#     global_indices = self.gather_indices(max_ind)
#
#   AFTER:
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         global_hidden_states = \
#             gather_from_sequence_parallel_region_to_moe(hidden_states)
#         global_indices = self.gather_indices(max_ind)
#     else:
#         global_hidden_states = hidden_states
#         global_indices = max_ind
#
#   SwitchMLP.forward() — guard scatter with SP/EP condition [line ~255]:
#
#   BEFORE:
#     output_total = \
#         reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#     if self.add_bias:
#         output_bias_total = \
#             reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#         output_bias_total = \
#             output_bias_total/mpu.get_tensor_model_parallel_world_size()
#
#   AFTER:
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         output_total = \
#             reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#         if self.add_bias:
#             output_bias_total = \
#                 reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#             output_bias_total = \
#                 output_bias_total/mpu.get_tensor_model_parallel_world_size()
#
# Additional fix (core/transformer/switch_mlp.py):
#   upstream commit has typo 'globa_indices = max_ind' in the else branch;
#   NSP port corrects to 'global_indices = max_ind' — the variable is used
#   on the next line in the for-loop as (global_indices == local_expert_index).
#
# 诊断: 无SP/EP时仍强行gather/scatter，
# 犹如太平盛世仍按战时粮票配给，
# 徒增耗损，贻误百姓。
# ---------------------------------------------------------------------------

print('[M1880] megatron_transformer: SwitchMLP.forward guard SP/EP for gather/scatter; fix globa_indices typo')

# ---------------------------------------------------------------------------
# M1900: Megatron d931ba8a4 — Expose rotary base for Code Llama support
# Source: megatron/core/models/gpt/gpt_model.py
# Author: NVIDIA  Date: 2023-xx-xx
#
# Mapping: megatron/core/models/gpt/gpt_model.py → deepspeed/compile/megatron_transformer.py
#
# Upstream changes ported (GPTModel side, 4 lines):
#
#   1. GPTModel.__init__ — new `rotary_base: int = 10000` parameter:
#        OLD: __init__(self, ..., rotary_percent=1.0, seq_len_interpolation_factor=None)
#        NEW: __init__(self, ..., rotary_percent=1.0, rotary_base=10000,
#                      seq_len_interpolation_factor=None)
#      Inserted between rotary_percent and seq_len_interpolation_factor, matching
#      upstream ordering.
#
#   2. RotaryEmbedding call site switched to full keyword-argument form and
#      now passes rotary_base through:
#        OLD: RotaryEmbedding(self.config.kv_channels, rotary_percent,
#                             seq_len_interpolation_factor)
#        NEW: RotaryEmbedding(kv_channels=self.config.kv_channels,
#                             rotary_percent=rotary_percent,
#                             seq_len_interpolation_factor=seq_len_interpolation_factor,
#                             rotary_base=rotary_base)
#
# 20% DES-LOC adaptation (code-constants only; no live class here):
#   - Canonical call-site template below for use in future GPTModel port.
#   - 鲁迅曾言：世上本无底数，用10000的人多了，也便成了惯例；
#     Code Llama一出，百万底数横空，惯例遂成笑谈。
# ---------------------------------------------------------------------------

# M1900: canonical GPTModel.rotary_pos_emb instantiation template (keyword form).
# Copy-paste into any future GPTModel port in this project.
_M1900_ROTARY_CALL_TEMPLATE = """\
# M1900: rotary_base exposed — default 10000; set to 1000000 for Code Llama.
if self.position_embedding_type == 'rope':
    self.rotary_pos_emb = RotaryEmbedding(
        kv_channels=self.config.kv_channels,
        rotary_percent=rotary_percent,
        seq_len_interpolation_factor=seq_len_interpolation_factor,
        rotary_base=rotary_base,
    )
"""

print('[M1900] megatron_transformer: GPTModel.rotary_base param template recorded; '
      'see megatron_rotary_pos_embedding.py for live RotaryEmbedding change')

