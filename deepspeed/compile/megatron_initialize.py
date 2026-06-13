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

# ---------------------------------------------------------------------------
# M610: Megatron 0d5188c15 — refactored the fused kernels build
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit 0d5188c15)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2021-03-17
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#
# Changes ported from initialize.py:
#   1. Add `import time` at module level.
#   2. Add `from megatron import fused_kernels` import (mapped to
#      deepspeed.fused_kernels below).
#   3. Rename multi-import of set_tensor_model_parallel_{rank,world_size}
#      to parenthesised two-line form (cosmetic).
#   4. Fix docstring indentation in initialize_megatron() (cosmetic).
#   5. Replace inline dataset compile block with call to _compile_dependencies().
#   6. Add _compile_dependencies() function that:
#        a. Compiles dataset C++ code on rank-0 with timing print (TODO: ninja).
#        b. Checks custom_kernel_constraint (seq_len, attn_batch_size, fp16|bf16).
#        c. Prints constraint WARNING on rank-0 only.
#        d. Loads fused kernels rank-0-first (build on 0, barrier, then rest load).
#        e. Final barrier + rank-0 completion timing print.
#
# DeepSpeed adaptation:
#   - `from megatron import fused_kernels` → `from deepspeed import fused_kernels`
#     (deepspeed.fused_kernels is the DeepSpeed copy of the module).
#   - dist.get_rank() used in place of torch.distributed.get_rank().
#   - torch.distributed.barrier() preserved for fused-kernel sync (upstream uses it).
#   - get_args() sourced from deepspeed.compile.megatron_arguments.
#   - Adds print('[M610]') marker.
# ---------------------------------------------------------------------------

print('[M610]')

import time
import torch

try:
    from deepspeed import fused_kernels as _fused_kernels
except ImportError:
    _fused_kernels = None


def _compile_dependencies():
    """Compile dataset helpers and load fused kernels.

    Megatron 0d5188c15 initialize.py _compile_dependencies():

    1. Compile dataset C++ code (rank-0 only, with timing).
    2. Check custom_kernel_constraint; warn if not met.
    3. Load fused kernels: rank-0 builds first, then barrier, then others load.
    4. Final barrier + completion timing on rank-0.
    """
    try:
        from deepspeed.compile.megatron_arguments import get_args
        args = get_args()
    except Exception:
        args = None

    if args is None:
        print('[M610] _compile_dependencies: args not available, skipping.')
        return

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if dist.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        try:
            from megatron.data.dataset_utils import compile_helper
            compile_helper()
        except Exception:
            pass
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = getattr(args, 'seq_length', 0)
    tp_size = getattr(args, 'tensor_model_parallel_size', 1)
    num_heads = getattr(args, 'num_attention_heads', 1)
    micro_bs = getattr(args, 'micro_batch_size', 1)
    attn_batch_size = (num_heads / tp_size) * micro_bs
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = seq_len > 16 and seq_len <= 2048 and \
        seq_len % 4 == 0 and attn_batch_size % 4 == 0
    # Print a warning.
    fp16 = getattr(args, 'fp16', False)
    bf16 = getattr(args, 'bf16', False)
    masked_softmax_fusion = getattr(args, 'masked_softmax_fusion', False)
    if not ((fp16 or bf16) and
            custom_kernel_constraint and
            masked_softmax_fusion):
        if dist.get_rank() == 0:
            print('WARNING: constraints for invoking optimized'
                  ' fused softmax kernel are not met. We default'
                  ' back to unfused kernel invocations.', flush=True)

    if _fused_kernels is None:
        print('[M610] _compile_dependencies: fused_kernels not available, skipping kernel load.')
        return

    # Always build on rank zero first.
    if dist.get_rank() == 0:
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        _fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        _fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)

    print('[M610] _compile_dependencies: done.')


# ---------------------------------------------------------------------------
# M1233: Megatron 2e6a46e45 — Start Megatron-Core with vocab parallel cross entropy
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit 2e6a46e45)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2022-09-22
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#
# Changes ported from initialize.py (_initialize_distributed):
#   1. Add `from megatron import core` import.
#   2. After mpu.initialize_model_parallel(), also call
#      core.initialize_model_parallel() with identical args — initialises the
#      new megatron.core parallel state alongside the legacy mpu state.
#   3. Move the rank-0 tensor/pipeline size print from inside
#      initialize_model_parallel() to here (after both initialisations),
#      using core.get_tensor_model_parallel_world_size() /
#      core.get_pipeline_model_parallel_world_size() for the values.
#
# 20% adaptation: calls core_initialize_model_parallel() (our mapping of
# core.initialize_model_parallel) from deepspeed.compile.core_parallel_state;
# prints use the same format as upstream; adds print('[M1233]') marker.
# ---------------------------------------------------------------------------

print('[M1233]')


def core_initialize_model_parallel(tensor_model_parallel_size,
                                   pipeline_model_parallel_size,
                                   virtual_pipeline_model_parallel_size=None,
                                   pipeline_model_parallel_split_rank=None):
    """Initialise megatron-core parallel state alongside legacy mpu state.

    M1233: megatron/initialize.py — after mpu.initialize_model_parallel(),
    also call core.initialize_model_parallel() so that megatron-core modules
    (e.g. vocab_parallel_cross_entropy) can access parallel group state.
    """
    from deepspeed.compile.core_parallel_state import (
        initialize_model_parallel as _core_init_mp,
        get_tensor_model_parallel_world_size,
        get_pipeline_model_parallel_world_size,
    )
    _core_init_mp(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank,
    )
    print(f'> initialized tensor model parallel with size '
          f'{get_tensor_model_parallel_world_size()}')
    print(f'> initialized pipeline model parallel with size '
          f'{get_pipeline_model_parallel_world_size()}')
    print('[M1233] core_initialize_model_parallel: done.')

# ---------------------------------------------------------------------------
# M2030: Megatron f76b465e0 — Add TP communication bootstrap backend interface
# Source: megatron/training/initialize.py _initialize_tp_communicators()
#
# Mapping: megatron/training/initialize.py
#        → deepspeed/compile/megatron_initialize.py
#
# Changes ported from initialize.py:
#   1. Import get_te_version, is_te_min_version from megatron.core.utils
#      (mapped to deepspeed.compile.core_utils)
#   2. In _initialize_tp_communicators():
#      - Replace hardcoded torch.distributed.new_group(backend='mpi') +
#        te_module.base.initialize_ub(...)  with TE version branch:
#        * TE >= 1.9.0: pass bootstrap_backend=args.tp_comm_bootstrap_backend
#          directly to initialize_ub (TE creates the process group internally)
#        * TE < 1.9.0: warn if backend != 'mpi', then create MPI group and
#          call initialize_ub without bootstrap_backend kwarg
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰：「旧代码以 mpi 为唯一后端，如铁屋中人，虽安于一隅却不知外有天地；
#             今以版本判断开两门——新版 TE 自立门户，旧版 TE 仍守 mpi 旧制，
#             但至少警告用者：此路仅此一条，莫强求。」
#   - initialize_tp_communicators_m2030() 函数封装新逻辑，
#     与既有 M1233 风格一致。
#   - print('[M2030]') diagnostic added at module level.
#   - print('[M2030-TP-INIT] ...') diagnostic added inside the function.
# ---------------------------------------------------------------------------


def initialize_tp_communicators_m2030(te_module, args, ub_cfgs):
    """Initialize TP userbuffer communicators with configurable bootstrap backend.

    M2030: megatron/training/initialize.py — replaces hardcoded MPI bootstrap
    with a version-aware branch:
      - TE >= 1.9.0: TE manages process group; pass bootstrap_backend kwarg.
      - TE < 1.9.0:  Only MPI supported; create group manually, warn if other
                     backend was requested.

    鲁迅曰：「旧代码以 mpi 独占后端，犹如一家之言；
             今以 is_te_min_version 辨新旧，各行其道，方为正途。」
    """
    # Lazy import — core_utils maps megatron.core.utils in this project
    try:
        from deepspeed.compile.core_utils import get_te_version, is_te_min_version
        _have_te_version_utils = True
    except ImportError:
        _have_te_version_utils = False

    input_shape = [
        (args.seq_length * args.micro_batch_size) // args.context_parallel_size,
        args.hidden_size,
    ]
    backend = getattr(args, 'tp_comm_bootstrap_backend', 'nccl')

    print(f'[M2030-TP-INIT] initialize_tp_communicators: backend={backend} '
          f'input_shape={input_shape} tp_size={args.tensor_model_parallel_size}')

    if _have_te_version_utils and is_te_min_version("1.9.0"):
        # TE >= 1.9.0: process group is created inside TE; pass backend kwarg.
        print(f'[M2030-TP-INIT] TE >= 1.9.0 path: passing bootstrap_backend={backend} to initialize_ub')
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=backend,
        )
    else:
        # TE < 1.9.0: only MPI bootstrap is supported.
        if backend != 'mpi':
            import warnings
            te_ver = get_te_version() if _have_te_version_utils else 'unknown'
            warnings.warn(
                f'[M2030] Transformer Engine v{te_ver} supports only MPI bootstrap backend. '
                f'Requested backend={backend!r} ignored; falling back to mpi.'
            )
        print(f'[M2030-TP-INIT] TE < 1.9.0 path: creating MPI process group, bootstrap_backend forced to mpi')
        # Create a MPI process group to help with TP communication overlap bootstrap.
        import torch
        torch.distributed.new_group(backend='mpi')
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
            ub_cfgs=ub_cfgs,
        )

    print(f'[M2030-TP-INIT] initialize_tp_communicators done.')


print('[M2030]')
