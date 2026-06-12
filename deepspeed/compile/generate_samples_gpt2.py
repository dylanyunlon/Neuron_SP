# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M447: Megatron 5c45db4a7 — Initial implementation of pipelined text
#       generation
# ===========================================================================
#
# Upstream source:
#   tools/generate_samples_gpt2.py
#   (NVIDIA/Megatron-LM commit 5c45db4a79e91f2fb5620120594aa1727d2f7b37)
#   Author: Jared Casper <jcasper@nvidia.com>  Date: 2020-12-09
#
# Mapping: tools/generate_samples_gpt2.py
#          → deepspeed/compile/generate_samples_gpt2.py
#
# Changes ported from upstream:
#
#   model_provider():
#     - Add 'from megatron import mpu' import.
#     - Add 'from megatron.model import GPT2ModelFirstStage,
#       GPT2ModelLastStage, GPT2ModelIntermediateStage' import.
#     - When get_pipeline_model_parallel_world_size() > 1, select the
#       appropriate stage model class (First / Last / Intermediate) instead
#       of always constructing the monolithic GPT2Model.
#     - Fall through to GPT2Model(parallel_output=False) when pipeline size
#       is 1 (existing single-GPU / tensor-MP path unchanged).
#
# DeepSpeed adaptation notes:
#   This file is a reference stub that records the upstream model_provider
#   delta.  The actual model classes (GPT2ModelFirstStage etc.) are not
#   yet mapped into DeepSpeed; a placeholder select_pipeline_stage_model()
#   function documents the selection logic so it can be wired up when the
#   model classes are ported.
# ===========================================================================

print('[M447]')


def select_pipeline_stage_model(mpu, model_classes, num_tokentypes=0):
    """Select the correct GPT-2 model class for this pipeline stage.

    Megatron 5c45db4a7 tools/generate_samples_gpt2.py model_provider():
      if get_pipeline_model_parallel_world_size() > 1:
          if is_pipeline_first_stage():
              model = GPT2ModelFirstStage(num_tokentypes=0)
          elif is_pipeline_last_stage():
              model = GPT2ModelLastStage(num_tokentypes=0, parallel_output=False)
          else:
              model = GPT2ModelIntermediateStage(num_tokentypes=0)
      else:
          model = GPT2Model(num_tokentypes=0, parallel_output=False)

    Args:
        mpu: mpu module with is_pipeline_first_stage(),
             is_pipeline_last_stage(), and
             get_pipeline_model_parallel_world_size().
        model_classes: dict with keys 'full', 'first', 'last', 'intermediate'
                       mapping to model constructors.
        num_tokentypes: passed through to constructors (default 0).

    Returns:
        Instantiated model for this pipeline stage.
    """
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if mpu.is_pipeline_first_stage():
            return model_classes['first'](num_tokentypes=num_tokentypes)
        elif mpu.is_pipeline_last_stage():
            return model_classes['last'](
                num_tokentypes=num_tokentypes, parallel_output=False)
        else:
            return model_classes['intermediate'](num_tokentypes=num_tokentypes)
    else:
        return model_classes['full'](
            num_tokentypes=num_tokentypes, parallel_output=False)
