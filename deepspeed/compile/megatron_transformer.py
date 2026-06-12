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
