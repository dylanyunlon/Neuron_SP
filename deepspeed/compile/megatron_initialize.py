# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M329: Megatron 9026b86d8 — Initialization fixes: allowing simple case like
#        pytest pass, also making apex optional
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit 9026b86d8)
# Author: Boris Fomitchev <bfomitchev@nvidia.com>  Date: 2020-07-22
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#          (project convention: megatron top-level init → deepspeed/compile/)
#
# Changes ported:
#   1. initialize_megatron(): add early-return guard via
#      mpu.model_parallel_is_initialized() — prevents double-init when
#      pytest (or any caller) invokes initialize_megatron with the same
#      args twice in the same process.
#
# 20% adaptation: uses deepspeed.comm / mpu_initialize instead of
# megatron.mpu directly; keep guard semantics identical; adds print marker.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M345: Megatron 5c04ceb31 — Implementing lazy parallel initialization
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit 5c04ceb31)
# Author: Boris Fomitchev <bfomitchev@nvidia.com>  Date: 2020-08-05
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#
# Changes ported:
#   1. initialize_megatron(): wrap _initialize_distributed + _set_random_seed
#      into an inner ddp_init() closure.
#   2. When args.lazy_mpu_init is True, call set_model_parallel_world_size /
#      set_model_parallel_rank immediately, then return ddp_init for the
#      external DDP manager to call later.
#   3. Otherwise (default path), call ddp_init() inline and return None.
#   4. Docstring updated: documents the optional continuation-function return.
#
# 20% adaptation: uses deepspeed.comm / mpu_initialize setters; lazy_mpu_init
# attribute checked with hasattr/getattr for safety; adds print marker.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M373: Megatron 6e433055d — fix for nemo: do not initialize mpu if it is
#        already initialized
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit 6e433055d)
# Author: mohammad <mshoeybi@nvidia.com>  Date: 2020-10-01
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#
# Changes ported:
#   1. _initialize_distributed(): guard mpu.initialize_model_parallel() with
#      mpu.model_parallel_is_initialized() check — if already initialised,
#      print a notice and skip; otherwise initialise as usual.
#
# 20% adaptation: guard applied inside ddp_init() around the
# initialize_model_parallel call; uses local model_parallel_is_initialized()
# helper (already present from M329); adds print marker.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# M519: Megatron f0232865f — call makefile every run so we recompile if the
#        code has changed
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit f0232865f)
# Author: mohammad <mshoeybi@nvidia.com>  Date: 2021-01-25
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#
# Changes ported:
#   1. initialize_megatron(): replace try/except lazy compile_helper import
#      with an unconditional rank-0 compile_helper() call followed by a
#      barrier — ensures C++ dataset helpers are recompiled whenever source
#      changes, rather than silently reusing a stale build.
#
# 20% adaptation: logic placed at end of ddp_init() after dist is confirmed
# initialised; import torch kept local to match surrounding style.
# ---------------------------------------------------------------------------
print('[M519]')
print('[M373]')
print('[M345]')
# M556: Megatron dd8890626 — Interleaved pipeline execution and code refactoring
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit dd8890626)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2020-12-12
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#
# Changes ported:
#   1. _initialize_distributed(): pass virtual_pipeline_model_parallel_size_
#      from args to initialize_model_parallel() as third positional arg.
#      In DeepSpeed mapping: read args.virtual_pipeline_model_parallel_size
#      inside ddp_init() and call init_virtual_pipeline_model_parallel()
#      from mpu_initialize when set.
#
# 20% adaptation: uses deepspeed.comm / mpu_initialize; lazy_mpu_init path
# left unchanged; adds print('[M556]').
print('[M556]')

import deepspeed.comm as dist
from deepspeed.compile.mpu_initialize import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
    set_model_parallel_rank,
    set_model_parallel_world_size,
)


def model_parallel_is_initialized():
    """Check whether model and data parallel groups are initialized.

    Megatron 9026b86d8 initialize.py — WAR to allow simple cases like
    pytest calling initialize_megatron with the same args twice.
    Returns True once the model-parallel comm group has been set up.
    """
    if not dist.is_initialized():
        return False
    try:
        # A world_size > 0 and a valid rank both indicate that the
        # model-parallel group has been initialised at least once.
        world_size = get_model_parallel_world_size()
        rank = get_model_parallel_rank()
        return world_size is not None and rank is not None
    except Exception:
        return False


def initialize_megatron(extra_args_provider=None,
                        args_defaults=None,
                        ignore_unknown_args=False,
                        allow_no_cuda=False,
                        args=None):
    """Set global variables, initialize distributed, and set random seeds.

    Megatron 9026b86d8 initialize.py — idempotent init: if the
    model-parallel group is already set up (e.g. a second pytest call)
    we return immediately without re-running expensive distributed setup.

    Megatron 5c04ceb31 initialize.py — lazy parallel initialization:
    when args.lazy_mpu_init is True, sets basic model-parallel globals and
    returns a ddp_init() closure for an external DDP manager to call later.
    Otherwise performs the full initialization inline and returns None.

    `allow_no_cuda` should not be set unless using megatron/deepspeed for
    CPU-only data processing. In general this arg should not be set unless
    you know what you are doing.

    Returns a function to finalize distributed env initialization
    (optionally, only when lazy_mpu_init is True).
    """
    if args_defaults is None:
        args_defaults = {}

    if not allow_no_cuda:
        import torch
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'DeepSpeed/Megatron requires CUDA.'

    # This is temporary WAR to make simple case like pytest calling with
    # same args twice.  Need to implement factory clean init.
    if model_parallel_is_initialized():
        return None

    # Resolve args namespace — callers may pass it explicitly or rely on
    # deepspeed global state.
    if args is None:
        try:
            import deepspeed
            args = getattr(deepspeed, '_ds_args', None)
        except Exception:
            args = None

    # torch.distributed initialization closure (M345).
    def ddp_init():
        # Pytorch distributed — delegate to deepspeed.comm initialisation.
        if not dist.is_initialized():
            dist.init_distributed()

        # M373: do not initialize mpu if it is already initialized.
        if model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            model_parallel_size = getattr(args, 'model_parallel_size', 1)
            set_model_parallel_world_size(model_parallel_size)
            set_model_parallel_rank(dist.get_rank() % model_parallel_size
                                    if dist.is_initialized() else 0)
            # M556: initialise virtual pipeline parallel if configured.
            virtual_pipeline_size = getattr(
                args, 'virtual_pipeline_model_parallel_size', None)
            if virtual_pipeline_size is not None:
                from deepspeed.compile.mpu_initialize import (
                    init_virtual_pipeline_model_parallel)
                init_virtual_pipeline_model_parallel(virtual_pipeline_size)
                print(f'[M556] ddp_init: virtual_pipeline_model_parallel_size='
                      f'{virtual_pipeline_size}')

        # M519: call makefile every run so we recompile if the code has changed.
        import torch
        if dist.get_rank() == 0:
            from megatron.data.dataset_utils import compile_helper
            compile_helper()
        # Simple barrier
        torch.distributed.barrier()
        print('[M519]')

        print('[M373] ddp_init: model parallel init guard applied, '
              f'world_size={get_model_parallel_world_size()}, '
              f'rank={get_model_parallel_rank()}')
        print('[M345] ddp_init: distributed initialised, '
              f'world_size={get_model_parallel_world_size()}, '
              f'rank={get_model_parallel_rank()}')

    # M345: lazy_mpu_init path — external DDP manager takes responsibility.
    lazy_mpu_init = getattr(args, 'lazy_mpu_init', None)
    if lazy_mpu_init:
        # Set only the basic model-parallel globals so that callers can
        # query rank/world_size before DDP is fully initialised.
        model_parallel_size = getattr(args, 'model_parallel_size', 1)
        rank = getattr(args, 'rank', 0)
        set_model_parallel_world_size(model_parallel_size)
        # Return function for external DDP manager to call when DDP is ready.
        set_model_parallel_rank(rank)
        return ddp_init
    else:
        # Megatron/DeepSpeed's own DDP. Do initialization right away.
        ddp_init()
        # No continuation function.
        return None
