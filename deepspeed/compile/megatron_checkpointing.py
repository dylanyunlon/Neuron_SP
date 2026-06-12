# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M512: Megatron 78066ab08 — Fixing merge_mp_partitions
# Source: megatron/checkpointing.py (NVIDIA/Megatron-LM commit 78066ab08)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2021-01-21
#
# Mapping: megatron/checkpointing.py → deepspeed/compile/megatron_checkpointing.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from checkpointing.py:
#
#   1. set_checkpoint_version(value):
#        BEFORE: assert _CHECKPOINT_VERSION is None, "checkpoint version already set"
#        AFTER:  if _CHECKPOINT_VERSION is not None:
#                    assert _CHECKPOINT_VERSION == value, \
#                        "checkpoint versions do not match"
#      Allows multiple callers to set the same version without error (needed
#      by merge_mp_partitions which loads N partition checkpoints sequentially).
#
#   2. save_checkpoint():
#        if torch.distributed.get_rank() == 0:  →  print_rank_0(...)
#        if mpu.get_data_parallel_rank() == 0:  →  if not dist.is_initialized()
#                                                      or mpu.get_data_parallel_rank() == 0:
#        torch.distributed.barrier()            →  if dist.is_initialized():
#                                                      dist.barrier()
#        (Allows save_checkpoint to be called without distributed init, as
#        required by the merge_mp_partitions main() script.)
#
#   3. load_checkpoint():
#        Same dist.is_initialized() guards for barrier() calls.
#        Converts remaining torch.distributed.get_rank() == 0 prints to
#        print_rank_0().
#
# 20% adaptation: standalone module with simplified state; uses Python
# logging-compatible print_rank_0; does not depend on full Megatron global
# state machinery (get_args / mpu).  Adds print('[M512]') marker.
# ---------------------------------------------------------------------------

print('[M512]')

import torch

_CHECKPOINT_VERSION = None


def print_rank_0(message):
    """Print only on rank 0 (or when distributed is not initialised)."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message, flush=True)


def set_checkpoint_version(value):
    """Set global checkpoint version; allow re-setting to the same value.

    Megatron 78066ab08 checkpointing.py set_checkpoint_version():
      BEFORE: assert _CHECKPOINT_VERSION is None, 'checkpoint version already set'
      AFTER:  if _CHECKPOINT_VERSION is not None:
                  assert _CHECKPOINT_VERSION == value, \
                      'checkpoint versions do not match'

    The relaxed check enables merge_mp_partitions to call load_checkpoint()
    for each rank partition without hitting the 'already set' assertion on
    the second and subsequent partitions.
    """
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, \
            f'checkpoint versions do not match: {_CHECKPOINT_VERSION} vs {value}'
    _CHECKPOINT_VERSION = value
    print(f'[M512] set_checkpoint_version: version={value}')


def get_checkpoint_version():
    """Return the global checkpoint version."""
    return _CHECKPOINT_VERSION


def _barrier_if_initialized():
    """Call torch.distributed.barrier() only when distributed is initialized.

    Megatron 78066ab08: guards every barrier() call with is_initialized() so
    that utilities like merge_mp_partitions can run without a process group.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_checkpoint_safe(iteration, save_path, state_dict_fn,
                          get_checkpoint_name_fn,
                          ensure_directory_exists_fn):
    """Save a checkpoint guarded for non-distributed (single-process) use.

    Megatron 78066ab08 checkpointing.py save_checkpoint() key changes:
      • print_rank_0() instead of if get_rank()==0: print(...)
      • if not dist.is_initialized() or mpu.get_data_parallel_rank() == 0:
        (replaces bare  if mpu.get_data_parallel_rank() == 0:)
      • barrier guards: if dist.is_initialized(): dist.barrier()

    Args:
        iteration: training iteration number.
        save_path: directory to save checkpoint.
        state_dict_fn: callable() → dict to save.
        get_checkpoint_name_fn: callable(save_path, iteration) → filename.
        ensure_directory_exists_fn: callable(filename).
    """
    print_rank_0(f'saving checkpoint at iteration {iteration:7d} to {save_path}')

    # Only data-parallel rank 0 saves (or if not distributed at all).
    save_this_rank = True
    if torch.distributed.is_initialized():
        try:
            from .mpu_initialize import get_data_parallel_rank
            save_this_rank = (get_data_parallel_rank() == 0)
        except (ImportError, AttributeError):
            save_this_rank = (torch.distributed.get_rank() == 0)

    if save_this_rank:
        checkpoint_name = get_checkpoint_name_fn(save_path, iteration)
        ensure_directory_exists_fn(checkpoint_name)
        state_dict = state_dict_fn()
        state_dict['iteration'] = iteration
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary).
    _barrier_if_initialized()

    print_rank_0(f'  successfully saved checkpoint at iteration {iteration:7d} '
                 f'to {save_path}')

    # Update latest iteration tracker.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tracker_filename = save_path.rstrip('/') + '/latest_checkpointed_iteration.txt'
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not strictly necessary but matches Megatron).
    _barrier_if_initialized()


def load_checkpoint_safe(load_path, checkpoint_name,
                          model_load_fn,
                          set_checkpoint_version_fn=None):
    """Load a checkpoint guarded for non-distributed (single-process) use.

    Megatron 78066ab08 checkpointing.py load_checkpoint() key changes:
      • print_rank_0() instead of if get_rank()==0: print(...)
      • if dist.is_initialized(): dist.barrier()  (replaces unconditional barrier)

    Args:
        load_path: directory from which the checkpoint was loaded.
        checkpoint_name: full path to the checkpoint file.
        model_load_fn: callable(state_dict) — loads model state.
        set_checkpoint_version_fn: optional callable(version) — registers
            checkpoint version; defaults to set_checkpoint_version().
    """
    if set_checkpoint_version_fn is None:
        set_checkpoint_version_fn = set_checkpoint_version

    print_rank_0(f' loading checkpoint from {load_path}')

    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except Exception as e:
        print(f'[M512] load_checkpoint_safe: failed to load {checkpoint_name}: {e}')
        raise

    # Set checkpoint version.
    if 'checkpoint_version' in state_dict:
        set_checkpoint_version_fn(state_dict['checkpoint_version'])

    # Load model.
    model_load_fn(state_dict)

    # Some utilities want to load without distributed being initialized.
    _barrier_if_initialized()

    print_rank_0(f'  successfully loaded checkpoint from {load_path}')

    return state_dict.get('iteration', 0)


# ===========================================================================
# M556: Megatron dd8890626 — Interleaved pipeline execution and code refactoring
# ===========================================================================
#
# Upstream source: megatron/checkpointing.py → deepspeed/compile/megatron_checkpointing.py
#
# Changes ported:
#
#   save_checkpoint():
#     1. Unwrap model list (torchDDP → .module) per element rather than once.
#     2. If len(model) == 1: state_dict['model'] = model[0].state_dict_for_save_checkpoint()
#        Else: for i, set mpu.set_virtual_pipeline_model_parallel_rank(i),
#              state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()
#
#   load_checkpoint():
#     1. Same per-element DDP unwrap.
#     2. If len(model) == 1: model[0].load_state_dict(state_dict['model'])
#        Else: for i, set_virtual_pipeline_model_parallel_rank(i),
#              model[i].load_state_dict(state_dict['model%d' % i])
#
# 20% adaptation: existing save/load_checkpoint_safe() use callbacks; the
# virtual-pipeline logic is factored into two new helpers —
# build_virtual_pipeline_state_dict() and load_virtual_pipeline_state_dict()
# — that callers invoke as part of their state_dict_fn / model_load_fn.
# Adds print('[M556]').
# ===========================================================================

print('[M556]')


def build_virtual_pipeline_state_dict(model_list):
    """Build state_dict entries for a list of model modules.

    Megatron dd8890626 checkpointing.py save_checkpoint():
      if len(model) == 1:
          state_dict['model'] = model[0].state_dict_for_save_checkpoint()
      else:
          for i in range(len(model)):
              mpu.set_virtual_pipeline_model_parallel_rank(i)
              state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()

    Args:
        model_list: list of model modules (each unwrapped from DDP/FP16 wrappers).

    Returns:
        dict mapping 'model' (single stage) or 'model0', 'model1', … (multi-stage)
        to the corresponding state_dict_for_save_checkpoint() results.
    """
    from deepspeed.compile.mpu_initialize import set_virtual_pipeline_model_parallel_rank
    result = {}
    if len(model_list) == 1:
        result['model'] = model_list[0].state_dict_for_save_checkpoint()
    else:
        for i, model_module in enumerate(model_list):
            set_virtual_pipeline_model_parallel_rank(i)
            result['model%d' % i] = model_module.state_dict_for_save_checkpoint()
    print(f'[M556] build_virtual_pipeline_state_dict: {len(model_list)} stage(s)')
    return result


def load_virtual_pipeline_state_dict(model_list, state_dict, strict=True):
    """Load state_dict entries into a list of model modules.

    Megatron dd8890626 checkpointing.py load_checkpoint():
      if len(model) == 1:
          model[0].load_state_dict(state_dict['model'], strict=strict)
      else:
          for i in range(len(model)):
              mpu.set_virtual_pipeline_model_parallel_rank(i)
              model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    Args:
        model_list: list of model modules (each unwrapped from DDP/FP16 wrappers).
        state_dict: checkpoint state dict as loaded from disk.
        strict: passed through to load_state_dict().
    """
    from deepspeed.compile.mpu_initialize import set_virtual_pipeline_model_parallel_rank
    if len(model_list) == 1:
        model_list[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i, model_module in enumerate(model_list):
            set_virtual_pipeline_model_parallel_rank(i)
            model_module.load_state_dict(state_dict['model%d' % i], strict=strict)
    print(f'[M556] load_virtual_pipeline_state_dict: {len(model_list)} stage(s)')



# ===========================================================================
# M1203: Megatron 977efdfb9 — added backwards compatibility to checkpointing.py
# ===========================================================================
#
# Upstream source: megatron/checkpointing.py → deepspeed/compile/megatron_checkpointing.py
#
# Changes ported from get_checkpoint_names():
#
#   BEFORE: always built common_path with both tensor and pipeline MP ranks:
#       common_path = os.path.join(
#           checkpoints_path, directory,
#           "mp_rank_%02d_%03d" % (
#               mpu.get_tensor_model_parallel_rank(),
#               mpu.get_pipeline_model_parallel_rank()))
#       model_name = os.path.join(common_path, "model_rng.pt")
#       if use_distributed_optimizer:
#           optim_name = os.path.join(
#               common_path + "_%03d" % mpu.get_data_parallel_rank(), "optim.pt")
#       else:
#           optim_name = os.path.join(common_path, "optim.pt")
#
#   AFTER: when pipeline world size == 1, omit the pipeline rank suffix
#       (backward-compatible with checkpoints saved without pipeline parallelism);
#       also unified model/optim naming: non-distributed-optimizer now uses
#       "model_optim_rng.pt" for both names instead of separate files.
#
#       if mpu.get_pipeline_model_parallel_world_size() == 1:
#           common_path = os.path.join(checkpoints_path, directory,
#                                      'mp_rank_{:02d}'.format(
#                                          mpu.get_tensor_model_parallel_rank()))
#       else:
#           common_path = os.path.join(checkpoints_path, directory,
#                                      'mp_rank_{:02d}_{:03d}'.format(
#                                          mpu.get_tensor_model_parallel_rank(),
#                                          mpu.get_pipeline_model_parallel_rank()))
#       if use_distributed_optimizer:
#           model_name = os.path.join(common_path, "model_rng.pt")
#           optim_name = os.path.join(
#               common_path + "_%03d" % mpu.get_data_parallel_rank(), "optim.pt")
#       else:
#           model_name = optim_name = os.path.join(common_path, "model_optim_rng.pt")
#
# 20% adaptation: mpu calls are forwarded to deepspeed.compile.mpu_initialize
# equivalents; function is self-contained and usable independently of the
# Megatron global state machinery.  Adds print('[M1203]').
# ===========================================================================

print('[M1203]')


def get_checkpoint_names(checkpoints_path, iteration, use_distributed_optimizer,
                         release=False):
    """Return (model_name, optim_name) for the checkpoint at *iteration*.

    Megatron 977efdfb9 checkpointing.py get_checkpoint_names():
      Adds backwards-compatible path construction: when pipeline-model-parallel
      world size is 1, the directory name uses only the tensor-parallel rank
      (``mp_rank_XX``) instead of ``mp_rank_XX_YYY``, matching checkpoints
      saved by older code that predates pipeline parallelism support.

      Also unifies the non-distributed-optimizer file name to
      ``model_optim_rng.pt`` (combining the previous separate ``model_rng.pt``
      and ``optim.pt`` into a single file).

    Args:
        checkpoints_path: root directory that contains iteration sub-dirs.
        iteration: training iteration number (ignored when release=True).
        use_distributed_optimizer: if True, optimizer state is stored in a
            separate per-data-parallel-rank file.
        release: if True, uses the ``release`` sub-directory instead of an
            iteration-numbered one.

    Returns:
        (model_name, optim_name): absolute paths to the model and optimizer
        checkpoint files.
    """
    import os
    try:
        from deepspeed.compile import mpu_initialize as mpu
    except ImportError:
        # Fall back to a bare-minimum shim so the function is importable even
        # when the full mpu stack is not yet initialised.
        class _mpu_shim:
            @staticmethod
            def get_pipeline_model_parallel_world_size(): return 1
            @staticmethod
            def get_tensor_model_parallel_rank(): return 0
            @staticmethod
            def get_pipeline_model_parallel_rank(): return 0
            @staticmethod
            def get_data_parallel_rank(): return 0
        mpu = _mpu_shim()

    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    # Use both the tensor and pipeline MP rank.  If the pipeline world size is
    # 1 (no pipeline parallelism), omit the pipeline rank suffix for backwards
    # compatibility with checkpoints created by older Megatron versions.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        common_path = os.path.join(checkpoints_path, directory,
                                   'mp_rank_{:02d}'.format(
                                       mpu.get_tensor_model_parallel_rank()))
    else:
        common_path = os.path.join(checkpoints_path, directory,
                                   'mp_rank_{:02d}_{:03d}'.format(
                                       mpu.get_tensor_model_parallel_rank(),
                                       mpu.get_pipeline_model_parallel_rank()))

    # If using the distributed optimizer the optimizer's path must additionally
    # include the data parallel rank.
    if use_distributed_optimizer:
        model_name = os.path.join(common_path, 'model_rng.pt')
        optim_name = os.path.join(
            common_path + '_%03d' % mpu.get_data_parallel_rank(),
            'optim.pt')
    else:
        model_name = optim_name = os.path.join(common_path, 'model_optim_rng.pt')

    return model_name, optim_name

# M1204: Megatron b178e6fc5 — error fixes & tested
# ===========================================================================
#
# Upstream source: megatron/checkpointing.py → deepspeed/compile/megatron_checkpointing.py
#
# Changes ported from save_checkpoint():
#
#   1. Rename local variable: state_dict → model_state_dict (model/RNG block)
#      and state_dict → optim_state_dict (optimizer block).
#      Previously a single state_dict was reused for both blocks; now they are
#      kept as independent dicts so that model and optimizer can be saved to
#      separate files when use_distributed_optimizer is active.
#
#   2. Introduce save phase with use_distributed_optimizer branch:
#        if args.use_distributed_optimizer:
#            # Save model separate from optimizer.
#            if model_state_dict:
#                ensure_directory_exists(model_checkpoint_name)
#                torch.save(model_state_dict, model_checkpoint_name)
#            if optim_state_dict:
#                ensure_directory_exists(optim_checkpoint_name)
#                torch.save(optim_state_dict, optim_checkpoint_name)
#        else:
#            # Save model and optimizer together.
#            state_dict = {**model_state_dict, **optim_state_dict}
#            if state_dict:
#                ensure_directory_exists(model_checkpoint_name)
#                torch.save(state_dict, model_checkpoint_name)
#
#   3. Comment rewording: "Save args, model, RNG." → "Collect args, model, RNG."
#      and "Save optimizer state." → "Collect optimizer state."
#
# 20% adaptation: the split-save logic is expressed in
# save_checkpoint_distributed_optimizer() below, which extends the existing
# save_checkpoint_safe() callback pattern rather than modifying it in-place.
# Adds print('[M1204]').
# ===========================================================================

print('[M1204]')


def save_checkpoint_distributed_optimizer(
    iteration,
    save_path,
    model_state_dict,
    optim_state_dict,
    model_checkpoint_name,
    optim_checkpoint_name,
    ensure_directory_exists_fn,
    use_distributed_optimizer=False,
):
    """Save model and optimizer checkpoints, splitting files when use_distributed_optimizer.

    Megatron b178e6fc5 checkpointing.py save_checkpoint() save phase:

        if args.use_distributed_optimizer:
            # Save model separate from optimizer.
            if model_state_dict:
                ensure_directory_exists(model_checkpoint_name)
                torch.save(model_state_dict, model_checkpoint_name)
            if optim_state_dict:
                ensure_directory_exists(optim_checkpoint_name)
                torch.save(optim_state_dict, optim_checkpoint_name)
        else:
            # Save model and optimizer together.
            state_dict = {**model_state_dict, **optim_state_dict}
            if state_dict:  # only saves if populated
                ensure_directory_exists(model_checkpoint_name)
                torch.save(state_dict, model_checkpoint_name)

    Args:
        iteration: training iteration (for logging only).
        save_path: checkpoint directory (for logging only).
        model_state_dict: dict containing model/RNG/args entries (may be empty
            when this rank does not own data-parallel rank 0).
        optim_state_dict: dict containing optimizer/scheduler entries (may be
            empty when no_save_optim is set or rank is excluded).
        model_checkpoint_name: full path for the model checkpoint file.
        optim_checkpoint_name: full path for the optimizer checkpoint file
            (only used when use_distributed_optimizer=True).
        ensure_directory_exists_fn: callable(filename) — creates parent dirs.
        use_distributed_optimizer: when True, save model and optimizer to
            separate files; when False, merge them into model_checkpoint_name.
    """
    print_rank_0(
        f'[M1204] saving checkpoint at iteration {iteration:7d} to {save_path} '
        f'(use_distributed_optimizer={use_distributed_optimizer})'
    )

    if use_distributed_optimizer:
        # Collect args, model, RNG → model file.
        # Collect optimizer state → optim file.
        if model_state_dict:
            ensure_directory_exists_fn(model_checkpoint_name)
            torch.save(model_state_dict, model_checkpoint_name)
            print_rank_0(f'[M1204]   saved model  → {model_checkpoint_name}')
        if optim_state_dict:
            ensure_directory_exists_fn(optim_checkpoint_name)
            torch.save(optim_state_dict, optim_checkpoint_name)
            print_rank_0(f'[M1204]   saved optim  → {optim_checkpoint_name}')
    else:
        # Save model and optimizer together into model_checkpoint_name.
        state_dict = {**model_state_dict, **optim_state_dict}
        if state_dict:  # only saves if populated (inherits conditions above)
            ensure_directory_exists_fn(model_checkpoint_name)
            torch.save(state_dict, model_checkpoint_name)
            print_rank_0(f'[M1204]   saved combined → {model_checkpoint_name}')

    print_rank_0(f'[M1204] checkpoint save complete at iteration {iteration:7d}')

# ---------------------------------------------------------------------------
# M1278: Megatron d48d95ab8 — Open sourcing lm detoxification code
# Source: megatron/checkpointing.py (NVIDIA/Megatron-LM commit d48d95ab8)
# Author: Boxin Wang <boxinw@nvidia.com>  Date: 2022-11-23
#
# Mapping: megatron/checkpointing.py load_checkpoint()
#        → deepspeed/compile/megatron_checkpointing.py
#
# Changes ported from checkpointing.py (load_checkpoint, ~lines 532-580):
#
#   1. Guard args-check with `not args.finetune`:
#        Before:  if 'args' in model_state_dict:
#        After:   if 'args' in model_state_dict and not args.finetune:
#      When fine-tuning we skip consumed_train/valid_samples and
#      check_checkpoint_args so the pretrained config does not override
#      the new fine-tuning config.
#
#   2. New else-branch for the optimizer load block:
#        else:
#            if args.fp16 and optimizer is not None:
#                optimizer.reload_model_params()
#      After a successful optimizer state load we need to sync the fp16
#      master weights with the loaded model params.
#
# Adaptation note: Neuron_SP's megatron_checkpointing.py provides helper
# functions (load_checkpoint_safe, get_checkpoint_names, …) rather than a
# full load_checkpoint() port.  The upstream intent is recorded here as a
# marker; when a full load_checkpoint() is ported these two changes MUST be
# included.
# ---------------------------------------------------------------------------

def _load_checkpoint_finetune_notes():
    """Marker for M1278 load_checkpoint finetune changes (not callable).

    Upstream diff (megatron/checkpointing.py load_checkpoint):

        # 1. skip arg-check during fine-tune
        - if 'args' in model_state_dict:
        + if 'args' in model_state_dict and not args.finetune:
              checkpoint_args = model_state_dict['args']
              check_checkpoint_args(checkpoint_args)
              ...

        # 2. reload fp16 master weights when optimizer IS loaded
          try:
              optimizer.load_state_dict(...)
          except:
              ...
              sys.exit()
        + else:
        +     if args.fp16 and optimizer is not None:
        +         optimizer.reload_model_params()
    """
    pass  # documentation-only function

print('[M1278]')
