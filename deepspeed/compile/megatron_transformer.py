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
