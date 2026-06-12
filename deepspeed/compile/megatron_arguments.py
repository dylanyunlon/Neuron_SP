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
