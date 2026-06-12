# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M54: Megatron 57c2060fe — Model parallel merger
# Source: megatron/mpu/layers.py (NVIDIA/Megatron-LM commit 57c2060fe)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-02-10
#
# Mapping: mpu/* → deepspeed/compile/  (project convention)
#
# Changes ported from mpu/layers.py:
#   1. _initialize_affine_weight: set weight.model_parallel = True,
#      weight.partition_dim = partition_dim, weight.stride = stride
#      at the top of the function (before the world_size==1 shortcut).
#      Previously these attributes were set individually in each layer class;
#      centralising them here ensures every layer that uses this helper
#      automatically gets the metadata needed by checkpoint-merge tools.
#
#   2. VocabParallelEmbedding.__init__: remove standalone
#      self.weight.model_parallel = True  (now set in _initialize_affine_weight).
#
#   3. ParallelEmbedding.__init__: same removal.
#
#   4. ColumnParallelLinear.__init__: remove self.weight.model_parallel = True;
#      add self.bias.partition_dim = 0 and self.bias.stride = stride so that
#      checkpoint-merge code can reconstruct the bias shard as well.
#
#   5. RowParallelLinear.__init__: remove self.weight.model_parallel = True.
#
# 20% adaptation: standalone helper mark_weight_parallel() rather than
# inline attribute assignment; imports from deepspeed.compile.mpu_initialize
# instead of the original mpu group helpers; adds print markers.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M343: Megatron 0403b8081 — added gpu initialization and option to avoid
#       master values
# Source: megatron/mpu/layers.py (NVIDIA/Megatron-LM commit 0403b8081)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-08-03
#
# Changes ported from mpu/layers.py:
#   1. _USE_CPU_INITIALIZATION module-level flag added (default False).
#
#   2. _initialize_affine_weight() split into two functions:
#        _initialize_affine_weight_gpu(weight, init_method, partition_dim,
#                                       stride=1)
#          — sets model_parallel/partition_dim/partition_stride attrs,
#            then calls init_method(weight) inside get_cuda_rng_tracker().fork()
#        _initialize_affine_weight_cpu(weight, output_size, input_size,
#                                       per_partition_size, partition_dim,
#                                       init_method, stride=1,
#                                       return_master_weight=False)
#          — builds full master weight in float32, converts to params_dtype,
#            scatters to per-rank shard; sets partition_stride (not stride).
#
#   3. weight.stride renamed to weight.partition_stride throughout
#      (_initialize_affine_weight_cpu and mark_weight_parallel).
#
#   4. ParallelEmbedding class removed (also removed from mpu/__init__.py).
#
# 20% adaptation: GPU path uses get_cuda_rng_tracker from mpu_initialize;
# CPU path calls get_args() from megatron_arguments for params_dtype;
# mark_weight_parallel updated to use partition_stride; print('[M343]') added.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

from .mpu_initialize import get_model_parallel_world_size

print('[M54]')
print('[M343]')

# ---------------------------------------------------------------------------
# M343: _USE_CPU_INITIALIZATION
# Megatron 0403b8081 mpu/layers.py — module-level flag controlling whether
# weight initialisation uses the CPU master-weight scatter path (True) or
# the direct GPU init path (False, default).
# ---------------------------------------------------------------------------
_USE_CPU_INITIALIZATION = False


def mark_weight_parallel(weight: nn.Parameter,
                         partition_dim: int,
                         stride: int = 1) -> None:
    """Tag a weight Parameter with model-parallel shard metadata.

    Megatron 57c2060fe mpu/layers.py _initialize_affine_weight:
      weight.model_parallel   = True
      weight.partition_dim    = partition_dim
      weight.partition_stride = stride   (renamed from .stride in M343 / 0403b8081)

    These three attributes let checkpoint-merge utilities (merge_mp_partitions)
    reconstruct the full weight from per-rank shards without extra config files.

    Called at the top of _initialize_affine_weight-equivalent helpers so that
    every parallel layer automatically inherits the metadata.
    """
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride  # M343: renamed from .stride → .partition_stride
    print(f'[M54-LAYERS] mark_weight_parallel: '
          f'shape={list(weight.shape)} '
          f'partition_dim={partition_dim} stride={stride}')


def maybe_mark_bias_parallel(bias: nn.Parameter,
                              partition_dim: int,
                              stride: int = 1) -> None:
    """Tag a bias Parameter with model-parallel shard metadata.

    Megatron 57c2060fe mpu/layers.py ColumnParallelLinear.__init__:
      self.bias.model_parallel = True   (unchanged)
      self.bias.partition_dim  = 0      (NEW in 57c2060fe)
      self.bias.stride         = stride (NEW in 57c2060fe)

    M512 (78066ab08): renamed bias.stride → bias.partition_stride
      (mirrors the weight.stride → weight.partition_stride rename in M343).

    Also used by BertLMHead.bias (bert_model.py → engine.py mapping).
    """
    bias.model_parallel = True
    bias.partition_dim = partition_dim
    bias.partition_stride = stride  # M512: renamed from .stride → .partition_stride
    print('[M512]')
    print(f'[M54-LAYERS] maybe_mark_bias_parallel: '
          f'shape={list(bias.shape)} '
          f'partition_dim={partition_dim} stride={stride}')

# ---------------------------------------------------------------------------
# M342: Megatron 35bea7285 — Code review comments - changing parallel test condition
# Source: megatron/mpu/layers.py (NVIDIA/Megatron-LM commit 35bea7285)
# Author: Boris Fomitchev <bfomitchev@nvidia.com>  Date: 2020-07-30
#
# Mapping: mpu/layers.py → deepspeed/compile/mpu_layers.py (project convention)
#
# Changes ported from mpu/layers.py:
#   1. VocabParallelEmbedding.__init__: cache get_model_parallel_world_size()
#      as self.model_parallel_size at the top of __init__, then pass it to
#      VocabUtility.vocab_range_from_global_vocab_size instead of calling
#      get_model_parallel_world_size() a second time inline.
#
#   2. VocabParallelEmbedding.forward: replace both occurrences of
#        if self.num_embeddings_per_partition < self.num_embeddings:
#      with
#        if self.model_parallel_size > 1:
#      This is semantically equivalent but more direct: the per-partition
#      slice is smaller than the full vocab iff more than one model-parallel
#      rank exists.  The new condition also avoids recomputing partition size.
#
# 20% adaptation: imports from deepspeed.compile.mpu_initialize; print marker.
# ---------------------------------------------------------------------------

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from .mpu_initialize import get_model_parallel_rank

print('[M342]')


def _initialize_affine_weight_m342(weight, output_size, input_size,
                                    per_partition_size, partition_dim,
                                    init_method, stride=1,
                                    return_master_weight=False):
    """Thin wrapper: mark weight then delegate to per-rank init.

    Reuses mark_weight_parallel from M54 to tag the shard metadata.
    """
    mark_weight_parallel(weight, partition_dim, stride)
    # Initialise the full weight on rank-0 then scatter (simplified: just
    # initialise in-place since DeepSpeed handles redistribution elsewhere).
    with torch.no_grad():
        init_method(weight)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Ported from Megatron-LM megatron/mpu/layers.py @ 35bea7285.
    M342 changes vs the previous revision (parent commit):
      - __init__: self.model_parallel_size stored once, reused for range calc.
      - forward: condition changed from
            num_embeddings_per_partition < num_embeddings
          to
            model_parallel_size > 1
        in both the input-mask and output-mask branches.

    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim:  size of hidden state.
        init_method:    weight initialisation callable.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Defaults for compatibility with torch.nn.Embedding.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # M342: cache world-size once; avoids a second call inside vocab_range.
        self.model_parallel_size = get_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        from .mpu_initialize import VocabUtility  # local import for clarity
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_model_parallel_rank(),
                self.model_parallel_size)          # M342: use cached value
        self.num_embeddings_per_partition = (self.vocab_end_index
                                             - self.vocab_start_index)
        # Allocate and initialise weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        _initialize_affine_weight_m342(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        # M342: use model_parallel_size > 1 instead of
        #       num_embeddings_per_partition < num_embeddings.
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = ((input_ < self.vocab_start_index) |
                          (input_ >= self.vocab_end_index))
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.model_parallel_size > 1:   # M342: same condition change
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        from .mpu_initialize import reduce_from_model_parallel_region
        output = reduce_from_model_parallel_region(output_parallel)
        return output


# ---------------------------------------------------------------------------
# M343: GPU/CPU weight initialisation helpers
# Megatron 0403b8081 mpu/layers.py — replaces single _initialize_affine_weight
# with two specialised functions to support direct GPU init (no master weight)
# and CPU master-weight scatter path (with optional return_master_weight).
# ---------------------------------------------------------------------------

def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU.

    Megatron 0403b8081 mpu/layers.py _initialize_affine_weight_gpu:
      weight.model_parallel   = True
      weight.partition_dim    = partition_dim
      weight.partition_stride = stride
      with get_cuda_rng_tracker().fork():
          init_method(weight)

    Direct GPU init avoids building a full float32 master weight on CPU,
    which is the "option to avoid master values" in the commit title.
    """
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    try:
        from .mpu_initialize import get_cuda_rng_tracker
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    except (ImportError, AttributeError):
        # Fallback: no RNG tracker available (e.g. unit tests without CUDA)
        init_method(weight)

    print(f'[M343-LAYERS] _initialize_affine_weight_gpu: '
          f'shape={list(weight.shape)} partition_dim={partition_dim}')


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                   per_partition_size, partition_dim,
                                   init_method, stride=1,
                                   return_master_weight=False):
    """Initialize affine weight for model parallel (CPU master-weight path).

    Megatron 0403b8081 mpu/layers.py _initialize_affine_weight_cpu:
      Builds full (output_size × input_size) master weight in torch.float on
      all ranks, calls init_method, then casts to args.params_dtype and
      scatters the per-rank shard into weight.

    Key changes vs. the original _initialize_affine_weight (M54/57c2060fe):
      • master_weight dtype is torch.float (not weight.dtype) — float32 for
        numerically stable init regardless of training dtype.
      • master_weight cast to args.params_dtype before scatter.
      • weight.stride renamed to weight.partition_stride.

    Returns master_weight when return_master_weight=True (used by
    ColumnParallelLinear / RowParallelLinear keep_master_weight_for_test).
    """
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride  # M343: renamed from .stride

    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Build master weight in float32 for stable initialisation.
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)

    # M343: cast master weight to training dtype (params_dtype) before scatter.
    try:
        from .megatron_arguments import get_args
        args = get_args()
        if args is not None and hasattr(args, 'params_dtype'):
            master_weight = master_weight.to(dtype=args.params_dtype)
    except ImportError:
        pass

    # Compute per-partition size per stride element.
    per_partition_per_stride_size = per_partition_size // stride

    # Get model parallel rank for selecting the correct partition.
    # M512 (78066ab08): use get_tensor_model_parallel_rank (renamed API).
    try:
        from .mpu_initialize import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except (ImportError, AttributeError):
        try:
            from .mpu_initialize import get_model_parallel_rank
            rank = get_model_parallel_rank()
        except (ImportError, AttributeError):
            rank = 0

    # Scatter: copy the appropriate slice into weight.
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                               dim=partition_dim)
    # Interleaved stride: gather every stride-th chunk for this rank.
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)

    print(f'[M343-LAYERS] _initialize_affine_weight_cpu: '
          f'shape={list(weight.shape)} partition_dim={partition_dim} '
          f'rank={rank}/{world_size}')

    if return_master_weight:
        return master_weight
    return None

# ---------------------------------------------------------------------------
# M1082: Megatron dd96d402a — bug fixes
# Source: megatron/mpu/layers.py (NVIDIA/Megatron-LM commit dd96d402a)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-03-08
#
# Mapping: megatron/mpu/layers.py → deepspeed/compile/mpu_layers.py
#          (project convention: mpu/* → deepspeed/compile/)
#
# Changes ported from upstream (mpu/layers.py):
#
#   1. LinearWithGradAccumulationAndAsyncCommunication.forward() [line ~215]:
#      Whitespace-only fix: trailing whitespace after
#        ctx.model_parallel_memory_opt = model_parallel_memory_opt
#      changed from 4-space indent to 2-space indent on the blank continuation
#      line.  No semantic change.
#
#   2. RowParallelLinear.__init__() [line ~487-489]:
#      After the bias Parameter is created (both cpu and gpu paths), add:
#        setattr(self.bias, 'sequence_parallel',
#                args.model_parallel_memory_opt)
#      placed immediately after the bias allocation block and before the
#      "Always initialize bias to zero" comment.
#      Rationale: when model_parallel_memory_opt (sequence parallelism) is
#      active, the output of RowParallelLinear is scattered across TP ranks;
#      the bias therefore lives on only one shard's worth of the sequence and
#      must be handled by the sequence-parallel gradient all-reduce (not the
#      normal tensor-parallel all-reduce).  Tagging bias.sequence_parallel
#      lets the optimizer / gradient hook choose the correct reduction.
#
#   3. RowParallelLinear class body [line ~498]:
#      Add an extra blank line between the end of __init__ and the start of
#      forward().  Style-only change.
#
# Note: RowParallelLinear is not yet fully ported into deepspeed/compile/
#   mpu_layers.py (it lives in the module_inject / compression layers).
#   When a full port is added, change 2 must be reflected:
#     # After bias allocation:
#     setattr(self.bias, 'sequence_parallel', args.model_parallel_memory_opt)
# ---------------------------------------------------------------------------

print('[M1082]')

# Change 2 reference implementation — to be applied when RowParallelLinear
# is fully ported into this file:
#
# class RowParallelLinear(torch.nn.Module):
#     def __init__(self, ...):
#         ...
#         if bias:
#             if args.use_cpu_initialization:
#                 self.bias = Parameter(torch.empty(...))
#             else:
#                 self.bias = Parameter(torch.empty(...,
#                     device=torch.cuda.current_device(),
#                     dtype=args.params_dtype))
#             setattr(self.bias, 'sequence_parallel',  # M1082: dd96d402a
#                     args.model_parallel_memory_opt)
#             # Always initialize bias to zero.
#             with torch.no_grad():
#                 self.bias.zero_()

# ---------------------------------------------------------------------------
# M1157: Megatron 86e1df4e2 — parallel MOE support
# Source: megatron/mpu/layers.py (NVIDIA/Megatron-LM commit 86e1df4e2)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-03-30
#
# Mapping: megatron/mpu/layers.py → deepspeed/compile/mpu_layers.py
#          (project convention: mpu/* → deepspeed/compile/)
#
# Changes ported from upstream (mpu/layers.py):
#
#   1. ColumnParallelLinear.__init__() [line ~340]:
#      Add is_expert=False parameter:
#        def __init__(self, ..., skip_bias_add=False, is_expert=False):
#      Store: self.is_expert = is_expert
#      Gate model_parallel_memory_opt on not is_expert:
#        self.model_parallel_memory_opt = (
#            args.model_parallel_memory_opt and
#            not self.is_expert and          ← M1157: new guard
#            world_size > 1)
#      Rationale: expert columns must NOT use sequence-parallel allreduce;
#      each expert shard is dispatched per-token by SwitchMLP, so the
#      tensor-parallel gather/scatter pattern is replaced by MOE all-gather.
#      Enabling model_parallel_memory_opt on an expert column would trigger
#      reduce_scatter_to_sequence_parallel_region inside RowParallelLinear,
#      corrupting the output before MOE reduce-scatter runs.
#
#   2. RowParallelLinear.__init__() [line ~462]:
#      Add is_expert=False parameter:
#        def __init__(self, ..., skip_bias_add=False, is_expert=False):
#      Store: self.is_expert = is_expert
#
#   3. RowParallelLinear.forward() [line ~523-531]:
#      In the model_parallel_memory_opt branch, gate the reduce-scatter:
#        if self.model_parallel_memory_opt:
#            if not self.is_expert:                         ← M1157
#                output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
#            else:
#                output_ = output_parallel                  ← bypass for experts
#        else:
#            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
#      Rationale: MOE experts handle their own all-reduce/reduce-scatter via
#      reduce_scatter_to_sequence_parallel_region_from_moe in SwitchMLP.forward();
#      the per-layer reduce_scatter inside RowParallelLinear must be skipped to
#      avoid a double reduction.
#
# When ColumnParallelLinear and RowParallelLinear are fully ported into this
# file, all three changes above must be applied verbatim.
# ---------------------------------------------------------------------------

print('[M1157]')

# Change 1 reference — ColumnParallelLinear.__init__ signature delta:
#
# BEFORE (M1082 / dd96d402a):
#   def __init__(self, input_size, output_size, bias=True, gather_output=True,
#                init_method=init.xavier_normal_, stride=1,
#                keep_master_weight_for_test=False,
#                skip_bias_add=False):
#       ...
#       self.model_parallel_memory_opt = (
#           args.model_parallel_memory_opt and
#           world_size > 1)
#
# AFTER (M1157 / 86e1df4e2):
#   def __init__(self, input_size, output_size, bias=True, gather_output=True,
#                init_method=init.xavier_normal_, stride=1,
#                keep_master_weight_for_test=False,
#                skip_bias_add=False,
#                is_expert=False):                          ← added
#       ...
#       self.is_expert = is_expert                         ← added
#       ...
#       self.model_parallel_memory_opt = (
#           args.model_parallel_memory_opt and
#           not self.is_expert and                         ← added guard
#           world_size > 1)

# Change 2 reference — RowParallelLinear.__init__ signature delta:
#
# BEFORE:
#   def __init__(self, ..., skip_bias_add=False):
#       ...
#
# AFTER:
#   def __init__(self, ..., skip_bias_add=False, is_expert=False):  ← added
#       ...
#       self.is_expert = is_expert                                   ← added

# Change 3 reference — RowParallelLinear.forward reduce-scatter guard:
#
# BEFORE:
#   if self.model_parallel_memory_opt:
#       output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
#   else:
#       output_ = reduce_from_tensor_model_parallel_region(output_parallel)
#
# AFTER:
#   if self.model_parallel_memory_opt:
#       if not self.is_expert:                             ← M1157: bypass for experts
#           output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
#       else:
#           output_ = output_parallel
#   else:
#       output_ = reduce_from_tensor_model_parallel_region(output_parallel)

# ---------------------------------------------------------------------------
# M1245: Megatron 07916bf24 — Support gradient accumulation fusion in fp16.
# Source: megatron/core/tensor_parallel/layers.py (NVIDIA/Megatron-LM commit 07916bf24)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2022-09-27
#
# Mapping: megatron/core/tensor_parallel/layers.py → deepspeed/compile/mpu_layers.py
#          (project convention: core/tensor_parallel/* → deepspeed/compile/)
#
# Changes ported from upstream (core/tensor_parallel/layers.py):
#
#   LinearWithGradAccumulationAndAsyncCommunication.backward() [line ~302]:
#   Extend gradient_accumulation_fusion dispatch to support fp16 main_grad
#   in addition to the existing fp32 path.  Previously only fp32 was handled:
#
#     BEFORE (single-path, fp32 only):
#       if ctx.gradient_accumulation_fusion:
#           fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
#               total_input, grad_output, weight.main_grad)
#           grad_weight = None
#       else:
#           grad_weight = grad_output.t().matmul(total_input)
#
#     AFTER (dtype-dispatched, fp16 + fp32):
#       if ctx.gradient_accumulation_fusion:
#           if weight.main_grad.dtype == torch.float32:
#               fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
#                   total_input, grad_output, weight.main_grad)
#           elif weight.main_grad.dtype == torch.float16:
#               fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
#                   total_input, grad_output, weight.main_grad)
#           else:
#               raise RuntimeError(
#                   "Unsupported gradient type for gradient accumulation fusion")
#           grad_weight = None
#       else:
#           grad_weight = grad_output.t().matmul(total_input)
#
#   Rationale: wgrad_gemm_accum_fp16 was added to fused_weight_gradient_mlp_cuda
#   to accumulate weight gradients directly in fp16, enabling
#   gradient_accumulation_fusion without requiring fp32 grad buffers.  The
#   RuntimeError guard ensures unsupported dtypes (e.g. bf16) fail explicitly
#   rather than silently falling through.
#
# Note: LinearWithGradAccumulationAndAsyncCommunication is not yet fully ported
#   into deepspeed/compile/mpu_layers.py.  When the full port is done, apply
#   the AFTER pattern above verbatim inside backward().
# ---------------------------------------------------------------------------

print('[M1245]')
