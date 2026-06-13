# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M343: Megatron 0403b8081 — added gpu initialization and option to avoid
#       master values
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 0403b8081)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-08-03
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from arguments.py:
#   1. import torch added at module level.
#   2. parse_args(): after dynamic_loss_scale block, add params_dtype
#      assignment:
#        args.params_dtype = torch.float
#        if args.fp16: args.params_dtype = torch.half
#        if args.rank == 0: print('using {} for parameters ...')
#
# 20% adaptation: deepspeed uses ds_config.fp16.enabled rather than
# argparse args.fp16; _GLOBAL_ARGS singleton pattern used for get_args();
# adds print('[M343]') marker.
# ---------------------------------------------------------------------------

print('[M343]')

import torch

_GLOBAL_ARGS = None


def get_args():
    """Return the global args object.

    Megatron 0403b8081 arguments.py — global accessor used by mpu/layers.py
    to retrieve params_dtype without passing args through every call site.
    """
    return _GLOBAL_ARGS


def set_args(args):
    """Set the global args object.

    Called once during initialize_megatron so that downstream modules
    (mpu/layers.py) can access params_dtype via get_args().
    """
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    print(f'[M343] set_args: params_dtype={getattr(args, "params_dtype", None)}')


def set_params_dtype(args):
    """Set args.params_dtype based on fp16 flag.

    Megatron 0403b8081 arguments.py parse_args():
      args.params_dtype = torch.float
      if args.fp16:
          args.params_dtype = torch.half
      if args.rank == 0:
          print('using {} for parameters ...'.format(args.params_dtype), flush=True)

    Called after dynamic_loss_scale is resolved.
    """
    args.params_dtype = torch.float
    if getattr(args, 'fp16', False):
        args.params_dtype = torch.half
    rank = getattr(args, 'rank', 0)
    if rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype), flush=True)
    print(f'[M343] set_params_dtype: params_dtype={args.params_dtype}')
    return args


# ---------------------------------------------------------------------------
# M409: Megatron 2d8de2968 — Throw exception if ring_exchange is not
#       available when pipeline_model_parallel_size > 1
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 2d8de296890b9c01)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2020-10-30
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Change ported from arguments.py parse_args():
#   After setting args.pipeline_model_parallel_size = min(...), add:
#     if args.pipeline_model_parallel_size > 1:
#         if "ring_exchange" not in dir(torch.distributed):
#             raise Exception('PyTorch with torch.distributed.ring_exchange
#                             needed to run pipeline MP!')
#
# DeepSpeed adaptation: surfaced as validate_pipeline_mp_ring_exchange(args)
# so it can be called from engine init after pipeline size is resolved.
# ---------------------------------------------------------------------------

print('[M409]')


def validate_pipeline_mp_ring_exchange(args):
    """Raise if pipeline MP > 1 but torch.distributed.ring_exchange is absent.

    Megatron 2d8de2968 arguments.py parse_args():
      if args.pipeline_model_parallel_size > 1:
          if "ring_exchange" not in dir(torch.distributed):
              raise Exception('PyTorch with torch.distributed.ring_exchange
                              needed to run pipeline MP!')

    Called after pipeline_model_parallel_size is finalised.
    """
    pipeline_mp_size = getattr(args, 'pipeline_model_parallel_size', 1)
    if pipeline_mp_size > 1:
        if 'ring_exchange' not in dir(torch.distributed):
            raise Exception(
                'PyTorch with torch.distributed.ring_exchange needed '
                'to run pipeline MP!')
    print('[M409] validate_pipeline_mp_ring_exchange: '
          f'pipeline_mp_size={pipeline_mp_size}, '
          f'ring_exchange_available={"ring_exchange" in dir(torch.distributed)}')

# ---------------------------------------------------------------------------
# M451: Megatron 2623551d7 — Nicer error messages for deprecated arguments
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 2623551d7712a3b9)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2020-12-10
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py:
#   1. parse_args(): add deprecated-args assertion block after TP/PP print:
#        assert args.batch_size is None, '--batch-size argument is no longer
#            valid, use --micro-batch-size instead'
#        del args.batch_size
#        assert args.warmup is None, '--warmup argument is no longer valid,
#            use --lr-warmup-fraction instead'
#        del args.warmup
#        assert args.model_parallel_size is None, '--model-parallel-size is
#            no longer valid, use --tensor-model-parallel-size instead'
#        del args.model_parallel_size
#   2. _add_training_args(): add --batch-size deprecated stub arg.
#   3. _add_learning_rate_args(): add --warmup deprecated stub arg.
#   4. _add_distributed_args(): add --model-parallel-size deprecated stub arg.
#
# DeepSpeed adaptation: surfaced as validate_deprecated_args(args) +
# add_deprecated_args(parser) helpers callable from compile/initialize.
# ---------------------------------------------------------------------------

print('[M451]')


def validate_deprecated_args(args):
    """Assert deprecated CLI args are not set; delete them from namespace.

    Megatron 2623551d7 arguments.py parse_args() — nicer error messages:
      assert args.batch_size is None, '--batch-size argument is no longer
          valid, use --micro-batch-size instead'
      del args.batch_size
      assert args.warmup is None, '--warmup argument is no longer valid,
          use --lr-warmup-fraction instead'
      del args.warmup
      assert args.model_parallel_size is None, '--model-parallel-size is no
          longer valid, use --tensor-model-parallel-size instead'
      del args.model_parallel_size

    Only acts on attributes that exist in the namespace (i.e., were
    registered via add_deprecated_args); safe to call when the deprecated
    stubs were not registered.
    """
    if hasattr(args, 'batch_size'):
        assert args.batch_size is None, \
            '--batch-size argument is no longer valid, use --micro-batch-size instead'
        del args.batch_size

    if hasattr(args, 'warmup'):
        assert args.warmup is None, \
            '--warmup argument is no longer valid, use --lr-warmup-fraction instead'
        del args.warmup

    if hasattr(args, 'model_parallel_size'):
        assert args.model_parallel_size is None, \
            '--model-parallel-size is no longer valid, use --tensor-model-parallel-size instead'
        del args.model_parallel_size

    print('[M451] validate_deprecated_args: deprecated args validated and removed')


def add_deprecated_args(parser):
    """Register deprecated argument stubs so users get a clear error message.

    Megatron 2623551d7 — three deprecated args added across helpers:

    _add_training_args():
      group.add_argument('--batch-size', type=int, default=None,
                         help='Old batch size parameter, do not use. '
                         'Use --micro-batch-size instead')

    _add_learning_rate_args():
      group.add_argument('--warmup', type=int, default=None,
                         help='Old lr warmup argument, do not use. Use one of
                         the --lr-warmup-* arguments above')

    _add_distributed_args():
      group.add_argument('--model-parallel-size', type=int, default=None,
                         help='Old model parallel argument, do not use. Use
                         --tensor-model-parallel-size instead.')

    Adds all three to a single 'Deprecated Arguments' group on parser.
    Call before parser.parse_args() so that validate_deprecated_args() can
    catch and reject any usage with a clear message.
    """
    group = parser.add_argument_group(title='deprecated arguments')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the '
                       '--lr-warmup-* arguments above')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    print('[M451] add_deprecated_args: deprecated argument stubs registered')
    return parser

# ---------------------------------------------------------------------------
# M512: Megatron 78066ab08 — Fixing merge_mp_partitions
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 78066ab08)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2021-01-20
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py:
#   1. parse_args(): move "Set input defaults" block BEFORE the
#      micro_batch_size assertion (was after consumed_*_samples init).
#      This ensures defaults are set before any assertions run against them.
#
#   2. _add_checkpointing_args():
#        --no-load-optim: add default=None
#        --no-load-rng:   add default=None
#      (So the "set input defaults" block can override them via defaults dict.)
#
#   3. _add_distributed_args():
#        --use-cpu-initialization: action='store_true' → type=bool, required=False
#      (Allows merge_mp_partitions to inject via defaults dict, not CLI flag.)
#
# DeepSpeed adaptation: exposed as helper functions callable from compile/init.
# ---------------------------------------------------------------------------

print('[M512]')


def patch_checkpointing_args(parser):
    """Re-register --no-load-optim and --no-load-rng with default=None.

    Megatron 78066ab08 _add_checkpointing_args():
      group.add_argument('--no-load-optim', action='store_true', default=None)
      group.add_argument('--no-load-rng',   action='store_true', default=None)
    """
    group = parser.add_argument_group(title='M512 checkpointing patches')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    print('[M512] patch_checkpointing_args: no-load-optim/rng with default=None')
    return parser


def patch_distributed_args(parser):
    """Re-register --use-cpu-initialization as type=bool, required=False.

    Megatron 78066ab08 _add_distributed_args():
      group.add_argument('--use-cpu-initialization', type=bool, required=False)
    """
    group = parser.add_argument_group(title='M512 distributed patches')
    group.add_argument('--use-cpu-initialization', type=bool, required=False,
                       help='If set, affine parallel weights initialization uses CPU')
    print('[M512] patch_distributed_args: use-cpu-initialization as type=bool')
    return parser


# ---------------------------------------------------------------------------
# M556: Megatron dd8890626 — Interleaved pipeline execution and code refactoring
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit dd8890626)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2020-12-12
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from arguments.py _add_distributed_args():
#   1. Add --virtual-pipeline-model-parallel-size argument (int, default=None):
#        Number of virtual pipeline stages in physical stage.
#      Inserted after --model-parallel-size deprecated stub.
#
# 20% adaptation: added as a standalone patch_virtual_pipeline_args() helper
# that callers invoke alongside other distributed arg patches; uses the same
# add_argument_group pattern as existing M512 helpers; adds print('[M556]').
# ---------------------------------------------------------------------------

print('[M556]')


def patch_virtual_pipeline_args(parser):
    """Register --virtual-pipeline-model-parallel-size argument.

    Megatron dd8890626 _add_distributed_args():
      group.add_argument('--virtual-pipeline-model-parallel-size', type=int,
                         default=None,
                         help='Number of virtual pipeline stages in physical stage.')
    """
    group = parser.add_argument_group(title='M556 virtual pipeline patches')
    group.add_argument('--virtual-pipeline-model-parallel-size', type=int,
                       default=None,
                       help='Number of virtual pipeline stages in physical stage.')
    print('[M556] patch_virtual_pipeline_args: --virtual-pipeline-model-parallel-size registered')
    return parser


def set_input_defaults_early(args, defaults):
    """Apply defaults dict BEFORE micro_batch_size and other assertions.

    Megatron 78066ab08 parse_args(): "Set input defaults" block moved BEFORE
    the micro_batch_size assertion so that defaults can override args checked
    by early assertions (e.g. no_load_optim, no_load_rng, use_cpu_initialization).

    Sets args.<key> = defaults[key] only when attribute is currently None.
    Emits WARNING when user explicitly provided a value differing from default.
    """
    rank = getattr(args, 'rank', 0)
    for key in defaults:
        if getattr(args, key, None) is not None:
            if rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                       flush=True)
        else:
            setattr(args, key, defaults[key])
    print(f'[M512] set_input_defaults_early: applied {len(defaults)} default(s)')
    return args

# ---------------------------------------------------------------------------
# M544: Megatron 78a3dc323 — fixed arguments
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 78a3dc323f9da3c4f)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2021-02-03
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py _add_biencoder_args():
#   Fix --report-topk-accuracies help string: double-quote → single-quote
#   Fix --retriever-score-scaling help string: double-quote → single-quote
#
#   Before (broken — double-quoted string containing unescaped single quotes):
#     help="Which top-k accuracies to report '(e.g. '1 5 20')"
#     help="Whether to scale retriever scores by inverse 'square root of hidden size"
#   After (correct — single-quoted multi-line string concatenation):
#     help='Which top-k accuracies to report ' '(e.g. '1 5 20')'
#     help='Whether to scale retriever scores by inverse ' 'square root of hidden size'
#
# DeepSpeed adaptation: surfaced as add_biencoder_args(parser) helper
# callable from compile/initialize to register biencoder CLI arguments.
# ---------------------------------------------------------------------------

print('[M544]')


def add_biencoder_args(parser):
    """Register biencoder argument group with corrected help strings.

    Megatron 78a3dc323 _add_biencoder_args() — fixed quote style on two args:

      group.add_argument('--report-topk-accuracies', nargs='+', type=int,
                          default=[], help='Which top-k accuracies to report '
                          '(e.g. 1 5 20)')
      group.add_argument('--retriever-score-scaling', action='store_true',
                         help='Whether to scale retriever scores by inverse '
                          'square root of hidden size')

    The original used double-quoted help strings which caused syntactic
    confusion with embedded single quotes; the fix uses single-quoted
    implicit string concatenation throughout _add_biencoder_args().
    """
    group = parser.add_argument_group(title='biencoder arguments')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')

    # training — fixed: double-quote → single-quote (Megatron 78a3dc323)
    group.add_argument('--report-topk-accuracies', nargs='+', type=int,
                       default=[], help='Which top-k accuracies to report '
                       '(e.g. 1 5 20)')
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                       'square root of hidden size')

    # faiss index  (--faiss-use-gpu / --faiss-match / --faiss-topk-retrievals
    #               moved to add_tasks_retriever_args per M611)
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')

    print('[M544] add_biencoder_args: biencoder arguments registered')
    return parser
# M559: Megatron e3e5ea892 — Compute tensor chunk size more cleanly, and add
#       assertion for global batch size
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit e3e5ea892)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2021-01-20
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#          megatron/p2p_communication.py → deepspeed/runtime/pipe/p2p.py
#            (p2p.py already uses ring-exchange via M464; scatter_gather
#             tensor_chunk_shape path is absent — no porting needed there)
#
# Changes ported from arguments.py:
#   1. Typo fix in world_size assertion message:
#        'pipeline paralle ' → 'pipeline parallel '
#   2. After `assert args.global_batch_size > 0`, add:
#        if args.virtual_pipeline_model_parallel_size is not None:
#            assert args.global_batch_size % args.pipeline_model_parallel_size == 0, \
#                'global batch size is not divisible by pipeline parallel size when ' \
#                'using interleaved schedule'
#
# Changes NOT ported (p2p_communication.py):
#   - `from functools import reduce` / `import operator` additions and the
#     rewrite of tensor_chunk_shape from explicit multiply to
#     reduce(operator.mul, tensor_shape, 1) — deepspeed/runtime/pipe/p2p.py
#     was restructured by M464 and no longer contains a scatter_gather
#     tensor_chunk_shape code path.
# ---------------------------------------------------------------------------

print('[M559]')


def validate_global_batch_size_interleaved(args):
    """Assert global_batch_size divisible by pipeline_model_parallel_size for interleaved schedule.

    Megatron e3e5ea892 arguments.py parse_args():
      assert args.global_batch_size > 0
      if args.virtual_pipeline_model_parallel_size is not None:
          assert args.global_batch_size % args.pipeline_model_parallel_size == 0, \
              'global batch size is not divisible by pipeline parallel size when '\
              'using interleaved schedule'

    Only enforced when virtual_pipeline_model_parallel_size is set (i.e.,
    the interleaved schedule is active).  Safe to call when the attribute
    is absent (treated as None).
    """
    global_batch_size = getattr(args, 'global_batch_size', None)
    virtual_pp_size = getattr(args, 'virtual_pipeline_model_parallel_size', None)
    pp_size = getattr(args, 'pipeline_model_parallel_size', 1)

    assert global_batch_size is not None and global_batch_size > 0, \
        'global_batch_size must be a positive integer'

    if virtual_pp_size is not None:
        assert global_batch_size % pp_size == 0, \
            'global batch size is not divisible by pipeline parallel size when ' \
            'using interleaved schedule'

    print(f'[M559] validate_global_batch_size_interleaved: '
          f'global_batch_size={global_batch_size}, '
          f'pipeline_model_parallel_size={pp_size}, '
          f'virtual_pipeline_model_parallel_size={virtual_pp_size}')

# ---------------------------------------------------------------------------
# M565: Megatron dcef90697 — Change argument to control the number of model
#       chunks in a stage
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit dcef906978d28d73)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2021-02-13
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py:
#   1. parse_args(): replace direct --virtual-pipeline-model-parallel-size
#      check with --num-layers-per-virtual-pipeline-stage derivation:
#        if args.num_layers_per_virtual_pipeline_stage is not None:
#            assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0
#            args.virtual_pipeline_model_parallel_size = (
#                (args.num_layers // args.pipeline_model_parallel_size)
#                // args.num_layers_per_virtual_pipeline_stage)
#            assert args.global_batch_size % args.pipeline_model_parallel_size == 0
#        else:
#            args.virtual_pipeline_model_parallel_size = None
#   2. _add_distributed_args(): replace --virtual-pipeline-model-parallel-size
#      with --num-layers-per-virtual-pipeline-stage.
#
# DeepSpeed adaptation: surfaced as resolve_virtual_pipeline_size(args) +
# add_virtual_pipeline_arg(parser) helpers callable from compile/initialize.
# ---------------------------------------------------------------------------

print('[M565]')


def resolve_virtual_pipeline_size(args):
    """Compute virtual_pipeline_model_parallel_size from num_layers_per_virtual_pipeline_stage.

    Megatron dcef90697 arguments.py parse_args():
      if args.num_layers_per_virtual_pipeline_stage is not None:
          assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0, \
              'number of layers is not divisible by number of layers per virtual ' \
              'pipeline stage'
          args.virtual_pipeline_model_parallel_size = \
              (args.num_layers // args.pipeline_model_parallel_size) // \
              args.num_layers_per_virtual_pipeline_stage
          assert args.global_batch_size % args.pipeline_model_parallel_size == 0, \
              'global batch size is not divisible by pipeline parallel size when ' \
              'using interleaved schedule'
      else:
          args.virtual_pipeline_model_parallel_size = None

    Call after pipeline_model_parallel_size and global_batch_size are resolved.
    """
    num_layers_per_stage = getattr(args, 'num_layers_per_virtual_pipeline_stage', None)
    if num_layers_per_stage is not None:
        num_layers = getattr(args, 'num_layers', None)
        pipeline_mp_size = getattr(args, 'pipeline_model_parallel_size', 1)
        assert num_layers % num_layers_per_stage == 0, \
            'number of layers is not divisible by number of layers per virtual ' \
            'pipeline stage'
        args.virtual_pipeline_model_parallel_size = (
            (num_layers // pipeline_mp_size) // num_layers_per_stage)
        global_batch_size = getattr(args, 'global_batch_size', None)
        assert global_batch_size % pipeline_mp_size == 0, \
            'global batch size is not divisible by pipeline parallel size when ' \
            'using interleaved schedule'
    else:
        args.virtual_pipeline_model_parallel_size = None
    print('[M565] resolve_virtual_pipeline_size: '
          f'virtual_pipeline_model_parallel_size='
          f'{args.virtual_pipeline_model_parallel_size}')
    return args


def add_virtual_pipeline_arg(parser):
    """Register --num-layers-per-virtual-pipeline-stage (replaces --virtual-pipeline-model-parallel-size).

    Megatron dcef90697 _add_distributed_args():
      group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                         help='Number of layers per virtual pipeline stage')

    Replaces the old --virtual-pipeline-model-parallel-size argument; the
    virtual_pipeline_model_parallel_size value is now derived by
    resolve_virtual_pipeline_size() rather than passed directly.
    """
    group = parser.add_argument_group(title='M565 virtual pipeline arguments')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    print('[M565] add_virtual_pipeline_arg: --num-layers-per-virtual-pipeline-stage registered')
    return parser

# ---------------------------------------------------------------------------
# M598: Megatron 0aff3629e — Update argument names and fix merge error
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 0aff3629e)
# Author: Rewon Child <rchild@nvidia.com>  Date: 2021-03-04
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Change ported from arguments.py _add_logging_args():
#   Before: group.add_argument('--log-zeros', action='store_true',
#               help='If set, calculate and log the number of zeros in gradient.')
#   After:  group.add_argument('--log-num-zeros-in-grad', action='store_true',
#               help='If set, calculate and log the number of zeros in gradient.')
#
# Companion renames in optimizer/__init__.py and optimizer/optimizer.py:
#   args.log_zeros → args.log_num_zeros_in_grad
#   self.log_zeros → self.log_num_zeros_in_grad
#   num_zeros      → num_zeros_in_grad  (local variable + return value)
#
# Also fixes a merge conflict in training.py train_step():
#   <<<<<<< HEAD:  update_successfull, grad_norm, num_zeros = optimizer.step()
#   >>>>>>> main:  update_successful, grad_norm = optimizer.step()
#   Resolved to:   update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
#
# DeepSpeed adaptation (documentation only — no functional code change):
#   DeepSpeed's engine does not currently use `log_zeros` / `log_num_zeros_in_grad`
#   as a direct argument; gradient-zero counting is handled internally in
#   deepspeed/runtime/zero/ and deepspeed/runtime/engine.py.
#   This block documents the upstream rename so future argument bridging
#   (args → DeepSpeed config) uses the correct canonical name.
# ---------------------------------------------------------------------------

print('[M598]')

# ---------------------------------------------------------------------------
# M610: Megatron 0d5188c15 — refactored the fused kernels build
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 0d5188c15)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2021-03-17
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py:
#   1. Remove `from megatron import fused_kernels` import at top of module.
#   2. Remove the custom-kernel constraints check block from parse_args():
#        seq_len / attn_batch_size / custom_kernel_constraint computation,
#        the WARNING print for unmet constraints, and the three
#        fused_kernels.load_* calls conditional on args.masked_softmax_fusion
#        and args.fp32_residual_connection.
#      All of this logic is now consolidated in initialize.py's
#      _compile_dependencies() (ported in megatron_initialize.py).
#
# DeepSpeed adaptation:
#   - In deepspeed the kernel constraint check and load call are performed
#     by _compile_dependencies() inside megatron_initialize.py, mirroring
#     the upstream move from arguments.py → initialize.py.
#   - No new helpers needed here; this file documents the removal.
#   - Adds print('[M610]') marker.
# ---------------------------------------------------------------------------

print('[M610]')

# ---------------------------------------------------------------------------
# M611: Megatron 43c9137b9 — Fixed based on review recommendation
# Source commit: 43c9137b94edcbaa2a9d1e3c671e938bac4cc937
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2021-03-18
#
# Mapping:
#   megatron/arguments.py _add_data_args()      → deepspeed/compile/megatron_arguments.py
#   megatron/arguments.py _add_biencoder_args() → deepspeed/compile/megatron_arguments.py
#   tasks/main.py get_tasks_args()              → add_tasks_retriever_args() (new, below)
#   tasks/orqa/evaluate_orqa.py                 → no equivalent (skip)
#   tasks/orqa/natural_questions/qa_utils.py    → license comment only (skip)
#   tasks/orqa/natural_questions/tokenizers.py  → license comment only (skip)
#
# Changes ported:
#   1. megatron/arguments.py _add_data_args(): remove --qa-data-dev / --qa-data-test
#      (those args belong to task evaluation, not core data loading).
#
#   2. megatron/arguments.py _add_biencoder_args(): remove --faiss-use-gpu,
#      --faiss-match, --faiss-topk-retrievals from the biencoder group.
#      In Neuron_SP only --faiss-use-gpu was present (--faiss-match and
#      --faiss-topk-retrievals were not yet ported); --faiss-use-gpu is now
#      removed from add_biencoder_args() above.
#
#   3. tasks/main.py get_tasks_args(): receive the moved args under a new
#      "Retriever args" and "Faiss arguments for retriever" comment block.
#      Surfaced here as add_tasks_retriever_args() for callers that need
#      task-specific retriever/faiss CLI flags.
#
# DeepSpeed adaptation:
#   - add_biencoder_args() already had only --faiss-use-gpu in its faiss
#     section; the comment line is updated and the add_argument removed.
#   - add_tasks_retriever_args() consolidates qa-data + faiss flags the same
#     way upstream tasks/main.py.get_tasks_args() does.
# ---------------------------------------------------------------------------

def add_tasks_retriever_args(parser):
    """Register task-level QA / faiss retriever arguments.

    Megatron 43c9137b9 tasks/main.py get_tasks_args() addition:

      # Retriever args
      group.add_argument('--qa-data-dev', type=str, default=None,
                         help='Path to the QA dataset dev file.')
      group.add_argument('--qa-data-test', type=str, default=None,
                         help='Path to the QA dataset test file.')

      # Faiss arguments for retriever
      group.add_argument('--faiss-use-gpu', action='store_true',
                         help='Whether create the FaissMIPSIndex on GPU')
      group.add_argument('--faiss-match', type=str, default='string',
                          choices=['regex', 'string'],
                          help="Answer matching logic type")
      group.add_argument('--faiss-topk-retrievals', type=int, default=100,
                         help='Number of blocks to use as top-k during retrieval')

    These args were previously scattered across _add_data_args() and
    _add_biencoder_args(); the review recommendation moved them to the
    task-evaluation level where they are actually consumed.
    """
    group = parser.add_argument_group(title='M611 task retriever / faiss arguments')

    # Retriever args
    group.add_argument('--qa-data-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa-data-test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    # Faiss arguments for retriever
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--faiss-match', type=str, default='string',
                       choices=['regex', 'string'],
                       help='Answer matching logic type')
    group.add_argument('--faiss-topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    print('[M611] add_tasks_retriever_args: qa-data + faiss args registered')
    return parser


print('[M611]')

# ---------------------------------------------------------------------------
# M616: Megatron 182841f7d — Make sure pipeline-model-parallel size is
#       greater than 2 for interleaved schedule
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 182841f7df79410)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2021-03-20
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Change ported from arguments.py parse_args():
#   if args.num_layers_per_virtual_pipeline_stage is not None:
#       assert args.pipeline_model_parallel_size > 2, \           ← NEW
#           'pipeline-model-parallel size should be greater than 2 with ' \
#           'interleaved schedule'
#       assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0
#       ...
#
# DeepSpeed adaptation: the new assertion is added to the existing
# validate_pipeline_mp_size_interleaved() helper so it can be called from
# compile/initialize after pipeline_model_parallel_size is resolved;
# resolve_virtual_pipeline_size() is unmodified (it contains the layers
# divisibility assertion but not the pp-size > 2 guard).
# ---------------------------------------------------------------------------

print('[M616]')


def validate_pipeline_mp_size_interleaved(args):
    """Assert pipeline_model_parallel_size > 2 when interleaved schedule is active.

    Megatron 182841f7d arguments.py parse_args():
      if args.num_layers_per_virtual_pipeline_stage is not None:
          assert args.pipeline_model_parallel_size > 2, \\
              'pipeline-model-parallel size should be greater than 2 with ' \\
              'interleaved schedule'

    Call after pipeline_model_parallel_size and
    num_layers_per_virtual_pipeline_stage are resolved; safe to call when
    the attribute is absent (treated as None, no assertion executed).
    """
    num_layers_per_stage = getattr(args, 'num_layers_per_virtual_pipeline_stage', None)
    if num_layers_per_stage is not None:
        pipeline_mp_size = getattr(args, 'pipeline_model_parallel_size', 1)
        assert pipeline_mp_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
    print('[M616] validate_pipeline_mp_size_interleaved: '
          f'num_layers_per_virtual_pipeline_stage={num_layers_per_stage}, '
          f'pipeline_model_parallel_size={getattr(args, "pipeline_model_parallel_size", 1)}')

# ---------------------------------------------------------------------------
# M612: Megatron 0fa7175f0 — Bfloat fused softmax + fused layer norm
# Source commit: 0fa7175f0936db7fbe303ca47b25fafca49ef032
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2021-03-19
#
# Mapping:
#   megatron/arguments.py           → deepspeed/compile/megatron_arguments.py
#   megatron/fused_kernels/__init__ → deepspeed/fused_kernels/__init__.py
#   megatron/model/__init__.py      → (no equivalent; LayerNorm import)
#   megatron/model/fused_layer_norm → (no equivalent in this tree)
#   megatron/model/fused_softmax.py → (no equivalent; FusedScaleMaskSoftmax)
#   megatron/model/transformer.py   → deepspeed/model_implementations/transformers/
#   megatron/optimizer/__init__     → (no equivalent)
#   megatron/training.py            → deepspeed/runtime/engine.py
#
# Changes ported:
#   1. megatron/arguments.py parse_args() bf16 block:
#      REMOVED: `assert not args.masked_softmax_fusion`
#      KEPT: `assert not args.bias_gelu_fusion`, `assert not args.bias_dropout_fusion`
#      Rationale: fused softmax now supports bfloat16 (DISPATCH_HALF_AND_BFLOAT
#      macro added in type_shim.h), so the restriction is lifted.
#
#   2. megatron/fused_kernels/__init__.py load():
#      Mixed-precision fused layer norm is now loaded unconditionally
#      (previously guarded by `if args.fp32_residual_connection`).
#      Ported to deepspeed/fused_kernels/__init__.py — guard removed.
#
#   3. megatron/model/__init__.py:
#      Removed `import_layernorm(fp32_residual_connection, bf16)` factory.
#      Now directly re-exports `MixedFusedLayerNorm as LayerNorm`.
#      No DeepSpeed equivalent to port (no import_layernorm in this tree).
#
#   4. megatron/model/fused_softmax.py FusedScaleMaskSoftmax:
#      Added `input_in_bf16` parameter; introduced `self.input_in_float16`
#      (= fp16 OR bf16). Dispatch and cast paths updated for bf16.
#      No DeepSpeed equivalent to port.
#
#   5. megatron/model/transformer.py ParallelTransformerLayer:
#      Removed 3× `if self.bf16 and self.fp32_residual_connection: .bfloat16()`
#      casts after layernorm (now unnecessary since MixedFusedLayerNorm
#      preserves dtype).  No DeepSpeed equivalent to port.
#
#   6. megatron/training.py get_model():
#      Removed the `for module_ in model_.modules()` block that called
#      `module_.float()` on LayerNorm params when bf16+fp32_residual was active.
#      No DeepSpeed equivalent to port.
#
# DeepSpeed adaptation:
#   - Only the fused_kernels/__init__.py and the arguments constraint change
#     have equivalents in this codebase.  The model/optimizer changes are
#     documented here for traceability.
# ---------------------------------------------------------------------------

print('[M612]')


# ---------------------------------------------------------------------------
# M702: Megatron 6f72a2851 — add dialog dataset and special tokens in tokenizer
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 6f72a2851)
# Author: zihanl <zihanl@nvidia.com>  Date: 2021-06-28
#
# Mapping: megatron/arguments.py _add_dialog_ctrl_args()
#          → deepspeed/compile/megatron_arguments.py add_dialog_ctrl_args()
#
# Changes ported:
#   1. New argument group "dialog control" with five arguments:
#      --train-module, --data-folder, --dataset-name, --max-seq-len,
#      --spec_toks (comma-separated special tokens for GPT2BPE tokenizer).
#
# DeepSpeed adaptation:
#   - Function renamed add_dialog_ctrl_args (public, no leading underscore)
#     to match the existing add_* naming convention in this file.
#   - No other logic changes; all defaults match upstream exactly.
# ---------------------------------------------------------------------------


def add_dialog_ctrl_args(parser):
    """Add dialog control arguments (M702: Megatron 6f72a2851)."""
    group = parser.add_argument_group(title='dialog control')

    group.add_argument('--train-module', type=str, default='',
                       help='either control module or dialogue model (control or dialog)')
    group.add_argument('--data-folder', type=str, default='',
                       help='data folder (path of the data folder)')
    group.add_argument('--dataset-name', type=str, default='',
                       help='dataset name (e.g., wizard_of_wikipedia)')
    group.add_argument('--max-seq-len', type=int, default=1024,
                       help='maximum sequence length')
    group.add_argument('--spec_toks', type=str, default='[SEP],[CTRL],[PAD]',
                       help='additional special tokens')

    return parser


print('[M702]')


# ---------------------------------------------------------------------------
# M749: Megatron 3bd2e9738 — added flag/logic for emptying unused memory
#
# Source: megatron/arguments.py  _add_distributed_args()
#
# Change summary:
#   Replaced the commented-out boolean --empty-unused-memory-each-iter flag
#   with a 3-choice integer argument (0/1/2):
#     0 = off (default)
#     1 = moderate  → torch.cuda.empty_cache() called each iteration
#     2 = aggressive → empty_cache() called + raises Exception (debug probe)
#
# Megatron diff (verbatim):
#   - # group.add_argument('--empty-unused-memory-each-iter', action='store_true',
#   - #                    help='Call torch.cuda.empty_cache() each iteration '
#   - #                    '(training and eval), to reduce fragmentation',
#   - #                    default=False)
#   + group.add_argument('--empty-unused-memory-each-iter', default=0, type=int,
#   +                    choices=[0, 1, 2],
#   +                    help='Call torch.cuda.empty_cache() each iteration '
#   +                    '(training and eval), to reduce fragmentation.'
#   +                    '0=off, 1=moderate, 2=aggressive.')
#
# DeepSpeed adaptation:
#   DeepSpeed does not replicate Megatron's _add_distributed_args() parser
#   verbatim.  The flag is registered here as a standalone patch function
#   add_empty_unused_memory_arg() that can be called after the base parser
#   is constructed, mirroring the pattern used for prior distributed-args
#   patches (M512, M598, M616, etc.).
# ---------------------------------------------------------------------------


def add_empty_unused_memory_arg(parser):
    """Register --empty-unused-memory-each-iter (M749).

    Megatron 3bd2e9738 replaced the old commented-out boolean flag with a
    3-choice integer:
      0 = off (default)
      1 = moderate  — torch.cuda.empty_cache() called each train/eval iter
      2 = aggressive — same as 1 but also raises an Exception (debug probe)
    """
    group = parser.add_argument_group(title='M749 memory management')
    group.add_argument(
        '--empty-unused-memory-each-iter',
        default=0,
        type=int,
        choices=[0, 1, 2],
        help='Call torch.cuda.empty_cache() each iteration '
             '(training and eval), to reduce fragmentation. '
             '0=off, 1=moderate, 2=aggressive.',
    )
    print('[M749] add_empty_unused_memory_arg: --empty-unused-memory-each-iter registered')
    return parser


print('[M749]')

# ---------------------------------------------------------------------------
# M771: Megatron cb5e611d7 — tested
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit cb5e611d7)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2021-08-22
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Change ported from arguments.py parse_args(), inside the
# distribute-checkpointed-activations block:
#   assert args.activations_checkpoint_method is not None, \
#       'for distribute-checkpointed-activations to work you '
#       'need to use a activation-checkpoint method '
#   + assert args.num_layers_per_virtual_pipeline_stage is None, \
#   +     'currently distrobuted checkpoint activations only supported for ' \
#   +     'nointerleaved pipeline parallelism'
#
# DeepSpeed adaptation: surfaced as validate_distributed_checkpointing(args)
# callable after pipeline and activation-checkpoint args are resolved.
# ---------------------------------------------------------------------------

print('[M771]')


def validate_distributed_checkpointing(args):
    """Assert distribute-checkpointed-activations prereqs are met.

    Megatron cb5e611d7 arguments.py parse_args():
      assert args.activations_checkpoint_method is not None, \\
          'for distribute-checkpointed-activations to work you ' \\
          'need to use a activation-checkpoint method '
      assert args.num_layers_per_virtual_pipeline_stage is None, \\
          'currently distrobuted checkpoint activations only supported for ' \\
          'nointerleaved pipeline parallelism'

    Call after distribute_saved_activations,
    activations_checkpoint_method, and
    num_layers_per_virtual_pipeline_stage are resolved.
    Safe to call when distribute_saved_activations is False or absent.
    """
    if not getattr(args, 'distribute_saved_activations', False):
        return
    assert getattr(args, 'activations_checkpoint_method', None) is not None, \
        'for distribute-checkpointed-activations to work you ' \
        'need to use a activation-checkpoint method '
    assert getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is None, \
        'currently distrobuted checkpoint activations only supported for ' \
        'nointerleaved pipeline parallelism'
    print('[M771] validate_distributed_checkpointing: passed')

# ---------------------------------------------------------------------------
# M1005: Megatron b93bef00d — comments, cleanup.
# Source: megatron/arguments.py, megatron/model/transformer.py,
#         megatron/mpu/initialize.py, megatron/p2p_communication.py,
#         megatron/schedules.py, megatron/training.py
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-01
#
# Changes in upstream:
#   arguments.py:
#     • Removes commented-out old virtual_pipeline_model_parallel_size block
#       (used pipeline_model_parallel_size instead of
#       transformer_pipeline_model_parallel_size) from parse_args().
#     • Removes commented-out lutil/pax debug block from parse_args().
#     • Improves --standalone-embed-stage help text: adds note that for T5
#       the flag currently only affects the encoder embedding.
#
#   model/transformer.py:
#     • Removes '# >>>' / '# <<<' markers around NoopTransformerLayer class.
#     • Rewrites NoopTransformerLayer docstring: explains standalone embed
#       stage creates zero-layer virtual ranks, that input==output causes
#       memory optimisation failures, and that clone() disconnects them.
#     • Removes several commented-out raise/pax debug blocks from
#       ParallelTransformer.__init__() and forward().
#     • Improves comment on num_layers==0 branch: clearer prose, removes
#       stale bullet list.
#
#   mpu/initialize.py:
#     • Removes live 'raise Exception("hi.")' from
#       set_pipeline_model_parallel_world_size() (was before the docstring).
#     • Removes commented-out 'raise Exception("hi.")' from
#       get_pipeline_model_parallel_world_size().
#     • get_num_layers(): removes several commented-out code blocks and pax
#       debug calls; adds explanatory comments for standalone_embed_stage
#       logic in both encoder/decoder and decoder-only paths; removes
#       '# args)' comment suffix from is_pipeline_stage_before_split() call.
#     • Removes '# >>>'/'# <<<' wrapping around
#       is_pipeline_stage_before_split() and is_pipeline_stage_after_split()
#       function definitions; removes assert isinstance debug guard and
#       commented-out rank adjustment inside both functions.
#
#   p2p_communication.py:
#     • Removes local 'def make_viewless_tensor(t)' closure and replaces
#       the two call sites with direct mpu.make_viewless_tensor() calls
#       (matching the original commented-out code, completing the >>>++<
#       switchover).
#
#   schedules.py:
#     • Removes commented-out pax debug block from get_forward_backward_func().
#     • Removes commented-out assert for transformer_pipeline_model_parallel_size
#       microbatch divisibility; keeps the pipeline_model_parallel_size assert.
#     • Removes live mpu.assert_viewless_tensor(output_tensor) calls from
#       forward_step() (two occurrences).
#     • Removes commented-out id(input_tensor)==id(output_tensor) debug
#       block from forward_backward_pipelining_with_interleaving().
#
#   training.py:
#     • Removes several commented-out pax/lutil debug blocks from pretrain(),
#       get_model(), setup_model_and_optimizer(), and
#       build_train_valid_test_data_iterators().
#     • Removes '# args)' comment suffix from is_pipeline_stage_before_split()
#       and is_pipeline_stage_after_split() call sites.
#
# DS mapping notes:
#   megatron/arguments.py          → deepspeed/compile/megatron_arguments.py
#   megatron/model/transformer.py  → no direct DS counterpart in this repo
#   megatron/mpu/initialize.py     → deepspeed/compile/mpu_initialize.py
#   megatron/p2p_communication.py  → deepspeed/compile/megatron_p2p_communication.py
#   megatron/schedules.py          → deepspeed/compile/megatron_schedules.py
#   megatron/training.py           → deepspeed/compile/megatron_training.py
#
# All six DS counterpart files were inspected. None contain the upstream
# debug scaffolding targeted by this commit:
#   • megatron_arguments.py: resolve_virtual_pipeline_size() already uses
#     transformer_pipeline_model_parallel_size; no lutil/pax blocks present.
#   • mpu_initialize.py: set/get_pipeline_model_parallel_world_size and
#     get_num_layers/is_pipeline_stage_*_split not yet ported to this file;
#     no raise Exception("hi.") / assert isinstance / pax blocks present.
#   • megatron_p2p_communication.py: _communicate() already calls
#     mpu.make_viewless_tensor() directly; no local closure present.
#   • megatron_schedules.py: M972 (Megatron 8fc5e3233) previously recorded
#     as already covering the schedules cleanup; file in clean state.
#   • megatron_training.py: only helper stubs present; full pretrain /
#     get_model / build_train_valid_test_data_iterators not ported.
#
# No code changes required in this repo. Marker added for log continuity.
# ---------------------------------------------------------------------------

print('[M1005]')

# ---------------------------------------------------------------------------
# M1038: Megatron 1cd3650dc — more minor fixes
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 1cd3650dc)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-02-01
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py parse_args():
#   After args.pipeline_model_parallel_size = min(...), add:
#     args.transformer_pipeline_model_parallel_size = (
#         args.pipeline_model_parallel_size - 1
#         if args.standalone_embedding_stage else
#         args.pipeline_model_parallel_size
#     )
#   virtual_pipeline_model_parallel_size derivation now uses
#     args.transformer_pipeline_model_parallel_size instead of
#     args.pipeline_model_parallel_size.
#
#   _add_distributed_args():
#     REMOVED: --deallocate-pipeline-outputs
#     ADDED:   --standalone-embedding-stage (action='store_true', default=False)
#       "If set, *input* embedding layer is placed on its own pipeline stage,
#        without any transformer layers. (For T5, this flag currently only
#        affects the encoder embedding.)"
#
# DS adaptation:
#   Surfaced as set_standalone_embedding_args(args) + add_standalone_embedding_arg(parser)
#   callable from compile/initialize, matching the existing patch_* / add_* convention.
# ---------------------------------------------------------------------------

print('[M1038]')


def set_standalone_embedding_args(args):
    """Compute transformer_pipeline_model_parallel_size from standalone_embedding_stage.

    Megatron 1cd3650dc arguments.py parse_args():
      args.transformer_pipeline_model_parallel_size = (
          args.pipeline_model_parallel_size - 1
          if args.standalone_embedding_stage else
          args.pipeline_model_parallel_size
      )

    Must be called after pipeline_model_parallel_size is resolved.
    Sets both transformer_pipeline_model_parallel_size and updates
    virtual_pipeline_model_parallel_size if num_layers_per_virtual_pipeline_stage
    is set (replicating the updated derivation).
    """
    pp_size = getattr(args, 'pipeline_model_parallel_size', 1)
    standalone = getattr(args, 'standalone_embedding_stage', False)
    args.transformer_pipeline_model_parallel_size = (
        pp_size - 1 if standalone else pp_size
    )
    # Re-derive virtual_pipeline_model_parallel_size if applicable
    # (M565 logic updated: uses transformer_pipeline_model_parallel_size)
    num_layers_per_stage = getattr(args, 'num_layers_per_virtual_pipeline_stage', None)
    if num_layers_per_stage is not None:
        num_layers = getattr(args, 'num_layers', None)
        t_pp_size = args.transformer_pipeline_model_parallel_size
        if num_layers is not None and t_pp_size > 0:
            assert num_layers % num_layers_per_stage == 0, \
                'number of layers is not divisible by number of layers per virtual pipeline stage'
            args.virtual_pipeline_model_parallel_size = (
                (num_layers // t_pp_size) // num_layers_per_stage
            )
    print('[M1038] set_standalone_embedding_args: '
          f'pipeline_model_parallel_size={pp_size}, '
          f'standalone_embedding_stage={standalone}, '
          f'transformer_pipeline_model_parallel_size='
          f'{args.transformer_pipeline_model_parallel_size}')
    return args


def add_standalone_embedding_arg(parser):
    """Register --standalone-embedding-stage argument.

    Megatron 1cd3650dc _add_distributed_args() — replaces
    --deallocate-pipeline-outputs:
      group.add_argument('--standalone-embedding-stage', action='store_true',
                         default=False,
                         help='If set, *input* embedding layer is placed on
                         its own pipeline stage, without any transformer layers.
                         (For T5, this flag currently only affects the encoder
                         embedding.)')
    """
    group = parser.add_argument_group(title='M1038 standalone embedding stage')
    group.add_argument(
        '--standalone-embedding-stage',
        action='store_true',
        default=False,
        help='If set, *input* embedding layer is placed on its own pipeline '
             'stage, without any transformer layers. (For T5, this flag '
             'currently only affects the encoder embedding.)',
    )
    print('[M1038] add_standalone_embedding_arg: --standalone-embedding-stage registered')
    return parser

# ---------------------------------------------------------------------------
# M1245: Megatron 07916bf24 — Support gradient accumulation fusion in fp16.
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 07916bf24)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2022-09-27
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from arguments.py (validate_args, lines 168-175):
#
#   Remove the else-branch that disabled gradient_accumulation_fusion when
#   accumulate_allreduce_grads_in_fp32 was False.  Previously the code was:
#
#     if args.accumulate_allreduce_grads_in_fp32:
#         assert args.DDP_impl == 'local'
#         assert args.use_contiguous_buffers_in_local_ddp
#     else:
#         if args.gradient_accumulation_fusion:
#             args.gradient_accumulation_fusion = False
#             if args.rank == 0:
#                 print('Gradient accumulation fusion to linear layer weight '
#                       'gradient computation is supported only with fp32 '
#                       'gradient accumulation. Setting gradient_accumulation_fusion '
#                       'to False', flush=True)
#
#   The else-branch (lines 171-178) is deleted entirely so that
#   gradient_accumulation_fusion can remain enabled in fp16 mode now that
#   wgrad_gemm_accum_fp16 is available (see M1245 in mpu_layers.py).
#
# Adaptation note: megatron_arguments.py does not contain a full validate_args
# implementation; this marker records the upstream intent.  When validate_args
# is ported, the else-branch above must NOT be included.
# ---------------------------------------------------------------------------

print('[M1245]')

# ---------------------------------------------------------------------------
# M1278: Megatron d48d95ab8 — Open sourcing lm detoxification code
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit d48d95ab8)
# Author: Boxin Wang <boxinw@nvidia.com>  Date: 2022-11-23
#
# Mapping: megatron/arguments.py _add_inference_args()
#        → deepspeed/compile/megatron_arguments.py add_max_tokens_to_oom_arg()
#
# Changes ported from arguments.py (_add_inference_args, ~line 363-370):
#
#   New argument added to inference group:
#     --max-tokens-to-oom  (int, default 12000)
#     Replaces the module-level MAX_TOKENS_TO_OOM constant in
#     text_generation/generation.py so that the threshold is configurable
#     at launch time rather than hard-coded.
#
# Adaptation note: Neuron_SP surfaces inference args as standalone helper
# functions.  The new function add_max_tokens_to_oom_arg() follows the same
# pattern as existing helpers in this file.
# ---------------------------------------------------------------------------

def add_max_tokens_to_oom_arg(parser):
    """Add --max-tokens-to-oom argument (Megatron d48d95ab8).

    Registers the inference-time OOM guard threshold that replaced the
    hard-coded ``MAX_TOKENS_TO_OOM = 12000`` constant in
    ``deepspeed/text_generation/generation.py``.
    """
    group = parser.add_argument_group(title='M1278 inference OOM guard')
    group.add_argument(
        '--max-tokens-to-oom',
        type=int,
        default=12000,
        help='Maximum number of tokens during inference; '
             'tokens here is # in prompt + # to generate. '
             'Allows us to throw an error before OOM crashes server.',
    )
    print('[M1278] add_max_tokens_to_oom_arg: --max-tokens-to-oom registered')
    return parser

print('[M1278]')

# ---------------------------------------------------------------------------
# M1333: Megatron 1e0e555c4 — merging rope to main
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 1e0e555c4)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2023-03-31
#
# Mapping: megatron/arguments.py _add_network_size_args()
#        → deepspeed/compile/megatron_arguments.py add_rotary_position_args()
#
# Changes ported from _add_network_size_args() (~line 509):
#
#   Three new arguments added after --max-position-embeddings:
#     --use-rotary-position-embeddings  (store_true)
#       Enable rotary positional embeddings (RoPE).
#     --rotary-percent  (float, default 1.0)
#       Fraction of head dimension to apply rotary encoding; < 1.0 gives
#       partial RoPE (Wang & Komatsuzaki et al. mesh-transformer-jax).
#     --no-position-embedding  (store_false → dest add_position_embedding)
#       Disable learned absolute position embedding so models can rely on
#       RoPE alone.
#
# Adaptation note: Neuron_SP surfaces argument groups as standalone helper
# functions.  add_rotary_position_args() follows the same pattern as
# add_max_tokens_to_oom_arg() introduced in M1278.
# ---------------------------------------------------------------------------


def add_rotary_position_args(parser):
    """Add RoPE-related arguments (Megatron 1e0e555c4).

    Registers three arguments introduced when rotary position embeddings were
    merged to Megatron main:

    * ``--use-rotary-position-embeddings`` — toggle RoPE on/off.
    * ``--rotary-percent`` — partial RoPE fraction (default 100 %).
    * ``--no-position-embedding`` — disables the learned absolute position
      embedding so the model can rely purely on RoPE.
    """
    group = parser.add_argument_group(title='M1333 rotary position embedding')
    group.add_argument(
        '--use-rotary-position-embeddings',
        action='store_true',
        help='Use rotary positional embeddings or not',
    )
    group.add_argument(
        '--rotary-percent',
        type=float,
        default=1.0,
        help='Percent of rotary dimension to use, default 100%%',
    )
    group.add_argument(
        '--no-position-embedding',
        action='store_false',
        dest='add_position_embedding',
        help='Disable position embedding.',
    )
    print('[M1333] add_rotary_position_args: RoPE args registered')
    return parser


print('[M1333]')

# ---------------------------------------------------------------------------
# M1359: Megatron 05b808ef7 — Expand on apply-layernorm-1p description a bit.
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 05b808ef7)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-05
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Change ported from arguments.py _add_network_size_args():
#   Before:
#     group.add_argument('--apply-layernorm-1p', action='store_true',
#                        help='Weight adjustment centered around zero.')
#   After:
#     group.add_argument('--apply-layernorm-1p', action='store_true',
#                        help='Adjust LayerNorm weights such that they are centered '
#                        'around zero. This improves numerical stability.')
#
# The help string is expanded from a terse single phrase to a two-sentence
# description clarifying both the mechanism (weights centered around zero)
# and the motivation (improved numerical stability).
#
# DeepSpeed adaptation: surfaced as add_network_size_args_1p(parser) that
# registers --apply-layernorm-1p with the expanded help text, following
# the existing add_* convention in this file.
# ---------------------------------------------------------------------------

print('[M1359]')


def add_network_size_args_1p(parser):
    """Register --apply-layernorm-1p with expanded help text.

    Megatron 05b808ef7 _add_network_size_args():
      group.add_argument('--apply-layernorm-1p', action='store_true',
                         help='Adjust LayerNorm weights such that they are centered '
                         'around zero. This improves numerical stability.')

    The prior help string ('Weight adjustment centered around zero.') is
    replaced with the expanded two-part description that makes the
    numerical-stability motivation explicit.
    """
    group = parser.add_argument_group(title='M1359 network size — layernorm 1p')
    group.add_argument(
        '--apply-layernorm-1p',
        action='store_true',
        help='Adjust LayerNorm weights such that they are centered '
             'around zero. This improves numerical stability.',
    )
    print('[M1359] add_network_size_args_1p: --apply-layernorm-1p registered')
    return parser
# M1379: Megatron 3e71ad9c6 — Exit on usage of --checkpoint-activations
#        because it defaults to full recomputation which is slow.
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 3e71ad9c6)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-26
#
# Mapping: megatron/arguments.py validate_args()
#        → deepspeed/compile/megatron_arguments.py validate_checkpoint_activations_arg()
#
# Changes ported from validate_args() (lines 102-108):
#
#   Before this commit, passing --checkpoint-activations silently fell back to
#   full recomputation by setting:
#     args.recompute_granularity = 'full'
#     args.recompute_method = 'uniform'
#   and printing a deprecation notice.
#
#   After this commit the silent fallback is removed entirely.  The function
#   now prints a clear error message telling users to switch to
#   --recompute-activations or the explicit --recompute-granularity /
#   --recompute-method flags, then calls exit() so training never proceeds
#   with the slow full-recompute default by accident.
#
#   Diff summary (megatron/arguments.py):
#     -        args.recompute_granularity = 'full'
#     -        args.recompute_method = 'uniform'
#              if args.rank == 0:
#     -            print('--checkpoint-activations is no longer valid, '
#     -                  'use --recompute-granularity and --recompute-method  instead. '
#     -                  'Defaulting to recompute-granularity=full and recompute-method=uniform.')
#     +            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
#     +                  'or, for more control, --recompute-granularity and --recompute-method.')
#     +        exit()
#
# Adaptation note: Neuron_SP surfaces validate_args logic as standalone
# helper functions.  validate_checkpoint_activations_arg() should be called
# immediately after argument parsing, before any training setup, so that
# misconfigured runs fail fast with a clear message rather than silently
# degrading to slow full recomputation.
# ---------------------------------------------------------------------------


def validate_checkpoint_activations_arg(args):
    """Exit if the deprecated --checkpoint-activations flag is present.

    ``--checkpoint-activations`` used to silently default to full activation
    recomputation (the slowest possible setting).  This function reproduces
    the post-3e71ad9c6 behaviour: print an actionable error message and call
    ``exit()`` so that jobs never accidentally run with full recomputation.

    Users should migrate to one of:

    * ``--recompute-activations``  (selective recompute, recommended)
    * ``--recompute-granularity`` + ``--recompute-method``  (explicit control)

    Args:
        args: parsed argument namespace (from ``parse_args()`` or equivalent).
    """
    if args.checkpoint_activations:
        if args.rank == 0:
            print(
                '--checkpoint-activations is no longer valid, use --recompute-activations, '
                'or, for more control, --recompute-granularity and --recompute-method.'
            )
        exit()
    del args.checkpoint_activations
    print('[M1379]')

print('[M1379]')

# ---------------------------------------------------------------------------
# M1420: Megatron 397d0b2eb — Split TransformerConfig into BaseConfig and
#        TransformerConfig, use BaseConfig for model parallel functions.
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 397d0b2eb)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-01
#
# Mapping: megatron/arguments.py core_config_from_args()
#        → deepspeed/compile/megatron_arguments.py core_config_from_args()
#
# New function introduced by this commit.  Translates a parsed argument
# namespace (args) into a TransformerConfig by:
#   1. Iterating over all dataclass fields of TransformerConfig.
#   2. Copying any field whose name exists as an attribute on args.
#   3. Applying three manual translations that differ between args and config:
#        persist_layer_norm   ← not args.no_persist_layer_norm
#        layernorm_zero_centered_gamma ← args.apply_layernorm_1p
#        deallocate_pipeline_outputs   ← True  (always on)
#
# The upstream version imports dataclasses and TransformerConfig at module
# top level; here we import inside the function to avoid circular imports
# with deepspeed/compile/__init__.py.
#
# 10% adaptation: TransformerConfig imported from local module; adds
# print('[M1420]') marker at call time.
# ---------------------------------------------------------------------------


def core_config_from_args(args):
    """Translate a parsed argument namespace to a TransformerConfig.

    Mirrors megatron/arguments.py core_config_from_args() introduced in
    NVIDIA/Megatron-LM commit 397d0b2eb (M1420).

    Args:
        args: Parsed argument namespace (from parse_args() or equivalent).

    Returns:
        TransformerConfig populated from args.
    """
    import dataclasses
    from deepspeed.compile.core_transformer_transformer_config import TransformerConfig

    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    # M1506: Megatron 4ef31451d — batch_p2p_comm is the logical inverse of
    # overlap_p2p_comm; when overlap is on, batch mode must be off so that
    # individual async isend/irecv calls can be tracked separately and waited
    # on out-of-order.  Derived here to keep both config flags consistent with
    # the CLI flag rather than letting each call-site compute the negation.
    kw_args['batch_p2p_comm'] = not getattr(args, 'overlap_p2p_comm', False)
    print('[M1420]')
    print('[M1506] core_config_from_args: batch_p2p_comm=%s (overlap_p2p_comm=%s)' % (
        kw_args['batch_p2p_comm'], getattr(args, 'overlap_p2p_comm', False)))
    return TransformerConfig(**kw_args)

print('[M1420]')

# ---------------------------------------------------------------------------
# M1501: Megatron f9283c5a8 — Add option to overlap p2p communication
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit f9283c5a8)
# Author: Mostofa Patwary <mpatwary@nvidia.com>
#
# Mapping: megatron/arguments.py _add_distributed_args()
#        → deepspeed/compile/megatron_arguments.py patch_overlap_p2p_args()
#
# Upstream adds --overlap-p2p-communication (store_true, dest=overlap_p2p_comm)
# to _add_distributed_args() after --num-layers-per-virtual-pipeline-stage.
#
# 20% adaptation: surfaced as a standalone patch_overlap_p2p_args() helper
# matching the existing pattern (patch_virtual_pipeline_args, etc.); adds
# print('[M1501]') marker at registration and call time.
# ---------------------------------------------------------------------------


def patch_overlap_p2p_args(parser):
    """Register --overlap-p2p-communication argument.

    Megatron f9283c5a8 _add_distributed_args():
      group.add_argument('--overlap-p2p-communication',
                         action='store_true',
                         help='overlap pipeline parallel communication with '
                              'forward and backward chunks',
                         dest='overlap_p2p_comm')

    DS mapping: standalone helper following the same group/print pattern as
    patch_virtual_pipeline_args (M556) and patch_num_layers_per_vp_stage (M1097).
    """
    group = parser.add_argument_group(title='M1501 overlap p2p patches')
    group.add_argument(
        '--overlap-p2p-communication',
        action='store_true',
        help='overlap pipeline parallel communication with forward and backward chunks',
        dest='overlap_p2p_comm',
    )
    print('[M1501] patch_overlap_p2p_args: --overlap-p2p-communication registered, '
          'sets args.overlap_p2p_comm=True when present')


print('[M1501]')

# ---------------------------------------------------------------------------
# M1503: Megatron 2c13d1f95 — Consistent arg names
# Source: megatron/core/pipeline_parallel/schedules.py + megatron/training.py
#         (NVIDIA/Megatron-LM commit 2c13d1f95)
#
# Upstream renames function parameters for consistency:
#   overlap_p2p_communication → overlap_p2p_comm   (3 function signatures)
#   batch_p2p_communication   → batch_p2p_comm     (3 function signatures)
# training.py call-site updated to pass keyword args with the short names.
#
# Mapping: megatron/core/pipeline_parallel/schedules.py param renames
#        → deepspeed/compile/megatron_arguments.py (this file) + REAL_GPU_BENCHMARK.py
#
# Neuron_SP already used the short form since M1501 (dest='overlap_p2p_comm').
# This commit confirms naming consistency: schedules.py now matches the dest=
# names declared here.  No functional change — pure rename.
#
# 20% adaptation: add verify_p2p_arg_names() diagnostic that checks a live
# argparse Namespace for the canonical short names and prints a M1503 marker;
# called at argument-parse time to surface any stale long-name kwargs early.
# ---------------------------------------------------------------------------


def verify_p2p_arg_names(args):
    """Verify p2p comm args use canonical short names (Megatron 2c13d1f95).

    Megatron 2c13d1f95 renamed function parameters across schedules.py:
      overlap_p2p_communication → overlap_p2p_comm
      batch_p2p_communication   → batch_p2p_comm

    Neuron_SP has used the short form since M1501.  This helper surfaces any
    stale long-name attributes on the Namespace (e.g. injected by third-party
    argument parsers) so they are caught at startup rather than silently ignored
    at the forward-backward call-site.
    """
    stale = [a for a in ('overlap_p2p_communication', 'batch_p2p_communication')
             if hasattr(args, a)]
    if stale:
        raise AttributeError(
            f'[M1503] Stale long-form p2p arg names detected: {stale}. '
            f'Use overlap_p2p_comm / batch_p2p_comm (Megatron 2c13d1f95).')
    overlap = getattr(args, 'overlap_p2p_comm', False)
    batch = getattr(args, 'batch_p2p_comm', not overlap)
    print(f'[M1503] verify_p2p_arg_names: overlap_p2p_comm={overlap} '
          f'batch_p2p_comm={batch} — names consistent with Megatron 2c13d1f95')


print('[M1503]')

# ---------------------------------------------------------------------------
# M1506: Megatron 4ef31451d — Fixes/cleanup from overlap p2p merge
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 4ef31451d)
#
# Mapping: megatron/arguments.py core_transformer_config_from_args()
#        → deepspeed/compile/megatron_arguments.py core_config_from_args()
#
# Change ported from arguments.py core_transformer_config_from_args():
#   After kw_args['pipeline_dtype'] = args.params_dtype, add:
#     kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
#
# The overlap flag is the primary control surface; batch_p2p_comm is its
# logical inverse.  Deriving it here ensures the two flags are always kept
# consistent when a TransformerConfig (or BaseConfig) is built from CLI args,
# rather than relying on each call-site to compute `not overlap_p2p_comm`.
#
# Companion changes across model_parallel_config.py and schedules.py:
#   • overlap_p2p_comm field added to ModelParallelConfig (default False)
#   • batch_p2p_comm default changed False → True
#   • batch_p2p_sync field added (default True) — cuda sync workaround
#   • Bare `overlap_p2p_comm` local references in schedules.py replaced by
#     config.overlap_p2p_comm (consistency + correct config lookup)
#   • Bare `deallocate_pipeline_outputs` replaced by
#     config.deallocate_pipeline_outputs (same reason)
#   • Non-interleaved schedule: ValueError for !batch_p2p_comm removed
#     (the restriction was overly strict; non-interleaved now tolerates
#      individual send/recv even without overlap, so the guard is gone)
#
# 20% Neuron_SP adaptation:
#   • core_config_from_args (M1420) updated with batch_p2p_comm derivation
#   • BaseConfig (core_base_config.py) and PipelineConfig (pipeline_config.py)
#     updated with overlap_p2p_comm, batch_p2p_comm=True, batch_p2p_sync fields
#   • __post__init__ in both config classes enforces the mutual-exclusion
#     invariant with a ValueError + print diagnostic
#   • megatron_schedules.py skipped — that file is an early port (M556/M971)
#     that predates the config-object plumbing; the bare-variable references
#     targeted by the upstream diff do not exist here
# ---------------------------------------------------------------------------

print('[M1506]')
