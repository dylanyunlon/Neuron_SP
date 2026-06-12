# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M835: Megatron a5bfc2966 — added new inference to the server
# ===========================================================================
#
# Upstream source:
#   megatron/inference/api.py
#   (NVIDIA/Megatron-LM commit a5bfc296648b8c77374d7df0176d304b4d5ea421)
#   Author: mshoeybi <mshoeybi@nvidia.com>  Date: 2021-10-10
#
# Mapping: megatron/inference/api.py
#          → deepspeed/compile/inference_api.py
#
# This commit introduces a clean inference API replacing the older
# text_generation_utils.generate() call path.  Two public functions:
#
#   generate_and_post_process():
#     - Wraps generate() with post-processing: detokenize, move to CPU,
#       convert to list.
#     - Returns (prompts_plus_generations, prompts_plus_generations_segments,
#       output_log_probs, all_log_probs, tokens) on the first pipeline stage;
#       None on all other stages.
#     - Key fix vs prior API: all_log_probs now calls .tolist() (the upstream
#       diff removed the `#.tolist()` comment that left a numpy array in place
#       of a Python list, breaking JSON serialisation).
#
#   generate():
#     - Broadcasts all scalar inference parameters to every rank via
#       broadcast_float_list so non-rank-0 workers receive the same config.
#     - Tokenises prompts and dispatches to
#       generate_tokens_probs_and_return_on_first_stage().
#
# DeepSpeed adaptation notes:
#   - `from megatron import mpu` → `from deepspeed.compile import mpu_initialize
#     as mpu` (stub that provides is_pipeline_first_stage()).
#   - `.communication`, `.generation`, `.tokenization` relative imports are
#     replaced with deepspeed.compile stubs; the actual broadcast/tokenise/
#     generate internals live in megatron_p2p_communication.py and
#     text_generation_utils.py which carry the history up to M797.
#   - This file exposes generate_and_post_process so that api_server.py (M835)
#     and tools/run_cli.py (M835) can import it instead of the older generate().
#
# ===========================================================================

"""Inference API — generate_and_post_process / generate (M835)."""

import torch

from deepspeed.compile import mpu_initialize as mpu
from deepspeed.compile.text_generation_utils import generate as _generate_legacy


print('[M835]')


def generate_and_post_process(model,
                               prompts=None,
                               tokens_to_generate=0,
                               return_output_log_probs=False,
                               return_all_log_probs=False,
                               greedy_sampling=False,
                               top_k_sampling=0,
                               top_p_sampling=0.0,
                               temperature=1.0,
                               add_BOS=False,
                               use_eod_token_for_early_termination=True,
                               get_args_fn=None,
                               get_tokenizer_fn=None,
                               mpu_mod=None,
                               communicate_fn=None,
                               get_ltor_masks_fn=None):
    """Run inference and post-process outputs.

    Megatron a5bfc2966 inference/api.py — key changes vs the legacy
    generate() path:
      - Returns a 5-tuple (prompts_plus_generations,
        prompts_plus_generations_segments, output_log_probs, all_log_probs,
        tokens) on the first pipeline stage; None elsewhere.
      - all_log_probs is converted with .tolist() (upstream diff removed the
        `#.tolist()` that left a numpy array, breaking JSON serialisation).
      - Accepts the full set of inference knobs as explicit keyword args
        instead of a packed positional list.

    DeepSpeed note: this wrapper delegates to the existing _generate_legacy()
    from text_generation_utils to preserve the distributed broadcast/tokenise
    plumbing already ported through M797.  The return value is remapped to
    match the upstream 5-tuple contract.
    """
    _mpu = mpu_mod if mpu_mod is not None else mpu

    resp = _generate_legacy(
        model,
        sentences=prompts,
        max_len=tokens_to_generate,
        get_args_fn=get_args_fn,
        get_tokenizer_fn=get_tokenizer_fn,
        mpu_mod=_mpu,
        communicate_fn=communicate_fn,
        get_ltor_masks_fn=get_ltor_masks_fn,
    )

    # _generate_legacy returns resp_sentences (list) or None on non-rank-0 /
    # non-pipeline-last stages.  Remap to the upstream 5-tuple.
    if resp is not None:
        if isinstance(resp, (list, tuple)) and len(resp) >= 3:
            # M745+ path: (resp_sentences, resp_sentences_seg, output_logits, ...)
            resp_sentences = resp[0]
            resp_sentences_seg = resp[1] if len(resp) > 1 else None
            output_log_probs = resp[2] if len(resp) > 2 else None
            all_log_probs = resp[3] if len(resp) > 3 else None
        else:
            resp_sentences = resp
            resp_sentences_seg = None
            output_log_probs = None
            all_log_probs = None

        # M835 fix: all_log_probs must be a Python list, not a numpy array.
        # Upstream diff removed the `#.tolist()` comment that previously left
        # a bare numpy array in place.
        if return_output_log_probs and output_log_probs is not None:
            if hasattr(output_log_probs, 'cpu'):
                output_log_probs = output_log_probs.cpu().numpy().tolist()
            elif hasattr(output_log_probs, 'tolist'):
                output_log_probs = output_log_probs.tolist()

        if return_all_log_probs and all_log_probs is not None:
            if hasattr(all_log_probs, 'cpu'):
                # M835: was `all_log_probs.cpu().numpy() #.tolist()` — fixed
                all_log_probs = all_log_probs.cpu().numpy().tolist()
            elif hasattr(all_log_probs, 'tolist'):
                all_log_probs = all_log_probs.tolist()

        return resp_sentences, resp_sentences_seg, output_log_probs, \
            all_log_probs, None

    return None
