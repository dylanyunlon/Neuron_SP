import logging
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M512: Megatron 78066ab08 — Fixing merge_mp_partitions
# Source: megatron/model/module.py (NVIDIA/Megatron-LM commit 78066ab08)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2021-01-21
#
# Mapping: megatron/model/module.py → deepspeed/compile/megatron_module.py
#          (project convention: megatron/model/* → deepspeed/compile/)
#
# Change ported from module.py MegatronModule.initialize_word_embeddings():
#
#   BEFORE: Raises Exception if share_word_embeddings is False.
#           Proceeds to tie word-embedding weights across pipeline stages.
#
#   AFTER:  Same Exception check, then:
#             if args.pipeline_model_parallel_size == 1:
#                 return
#           (Early return when there is only one pipeline stage, because
#           there is no "other end" to synchronise embeddings with.
#           merge_mp_partitions calls initialize_word_embeddings() even for
#           non-pipeline models; without this guard it would attempt to
#           broadcast across a non-existent pipeline group.)
#
# 20% adaptation: exposed as a standalone guard function rather than an
# inline conditional; callable from merge_mp_partitions and engine init.
# Adds  logging.debug('[M512]') marker.
# ---------------------------------------------------------------------------

logging.debug('[M512]')


def should_skip_word_embedding_init(args):
    """Return True when word-embedding cross-stage sync should be skipped.

    Megatron 78066ab08 model/module.py MegatronModule.initialize_word_embeddings():
      if args.pipeline_model_parallel_size == 1:
          return

    When pipeline_model_parallel_size == 1 there is only a single pipeline
    stage, so there is no separate "last stage" to sync embeddings with.
    Calling the full cross-stage tie logic in this case will attempt to
    broadcast across a pipeline group that does not exist, causing a hang or
    assertion failure.

    Args:
        args: namespace with pipeline_model_parallel_size attribute.

    Returns:
        True  → caller should return immediately (skip embedding sync).
        False → caller should proceed with cross-stage embedding tie.
    """
    pipeline_size = getattr(args, 'pipeline_model_parallel_size', 1)
    skip = (pipeline_size == 1)
    
    logging.debug(f'[M512] should_skip_word_embedding_init: '
          f'pipeline_size={pipeline_size} skip={skip}')
    return skip
