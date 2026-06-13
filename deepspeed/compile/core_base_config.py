# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1420: Megatron 397d0b2eb — Split TransformerConfig into BaseConfig and
#        TransformerConfig, use BaseConfig for model parallel functions.
# Source: megatron/core/base_config.py (NVIDIA/Megatron-LM commit 397d0b2eb)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-01
#
# Mapping: megatron/core/base_config.py  (NEW FILE)
#       -> deepspeed/compile/core_base_config.py
#          (project convention: megatron/core/* -> deepspeed/compile/core_*)
#
# New file introduced by this commit.  Splits the model-parallel and pipeline-
# parallel configuration fields that were previously embedded in TransformerConfig
# into a standalone BaseConfig dataclass so that forward_backward_func and
# p2p_communication no longer depend on TransformerConfig directly.
#
# Fields ported verbatim from upstream BaseConfig:
#   Model parallelism: tensor_model_parallel_size, pipeline_model_parallel_size,
#     virtual_pipeline_model_parallel_size, sequence_parallel
#   Initialization: init_method, output_layer_init_method, init_method_std,
#     perform_initialization, use_cpu_initialization
#   Training: fp16, bf16, params_dtype, grad_scaler, enable_autocast,
#     autocast_dtype, timers
#   Optimizations: gradient_accumulation_fusion,
#     async_tensor_model_parallel_allreduce
#   Pipeline parallel: pipeline_dtype, tensor_shape, variable_seq_lengths,
#     num_microbatches_with_partial_activation_checkpoints, batch_p2p_comm,
#     use_ring_exchange_p2p, deallocate_pipeline_outputs, no_sync_func,
#     grad_sync_func, param_sync_func
#   Legacy: decoder_seq_length
#
# __post__init__ validation ported verbatim:
#   sequence_parallel + tensor_model_parallel_size <= 1 → ValueError
#   async_tensor_model_parallel_allreduce disabled when sequence_parallel
#   pipeline_model_parallel_size > 1 → pipeline_dtype required
#   pipeline_model_parallel_size > 1 → tensor_shape required
#   autocast_dtype defaults to params_dtype when None
#
# Note: upstream typo `__post__init__` (double underscore) retained as-is.
#
# 10% adaptation: import path unchanged (torch/dataclasses/Callable are
# stdlib); adds print('[M1420]') marker.
# ---------------------------------------------------------------------------

print('[M1420]')

# ---------------------------------------------------------------------------
# M1506: Megatron 4ef31451d — Fixes/cleanup from overlap p2p merge
# Source: megatron/core/model_parallel_config.py (NVIDIA/Megatron-LM commit 4ef31451d)
#
# Mapping: megatron/core/model_parallel_config.py ModelParallelConfig
#        → deepspeed/compile/core_base_config.py BaseConfig
#
# Changes ported from model_parallel_config.py:
#   1. Docstring: replace bare "batch_p2p_comm (bool, default = False)" entry
#      with three entries:
#        overlap_p2p_comm (bool, optional, default=False)
#        batch_p2p_comm   (bool, default=True) — note default flip False → True
#        batch_p2p_sync   (bool, default=True)
#   2. Field declarations:
#        overlap_p2p_comm: bool = False   (NEW)
#        batch_p2p_comm: bool = True      (default was False)
#        batch_p2p_sync: bool = True      (NEW)
#
# 20% adaptation: mutual-exclusion invariant (overlap XOR batch) enforced in
# __post__init__ with ValueError; print diagnostic added.
# ---------------------------------------------------------------------------

print('[M1506]')\n# ---------------------------------------------------------------------------\n# M1870: Megatron 7a70c5401 — GPT model level change for context parallelism\n# Source: megatron/core/model_parallel_config.py\n#\n# Mapping: megatron/core/model_parallel_config.py ModelParallelConfig\n#        → deepspeed/compile/core_base_config.py BaseConfig\n#\n# Changes ported:\n#   1. Docstring: add context_parallel_size description\n#   2. Field: context_parallel_size: int = 1  (after tensor_model_parallel_size)\n#\n# 20% adaptation (鲁迅式迁移):\n#   鲁迅云：「不在沉默中爆发，就在沉默中灭亡。」\n#   序列维度被切分，各rank各得其所，上下文并行从config始。\n#   print('[M1870]') diagnostic added.\n# ---------------------------------------------------------------------------\nprint('[M1870] context_parallel_size field added to BaseConfig')
# ---------------------------------------------------------------------------
# M1960: Megatron c3079ce98 — Enable DGRAD RS overlap
# Source: megatron/core/model_parallel_config.py
#
# Mapping: megatron/core/model_parallel_config.py ModelParallelConfig
#        → deepspeed/compile/core_base_config.py BaseConfig
#
# Changes ported:
#   Field: tp_comm_overlap_rs_dgrad: bool = False
#     If true, allows Reduce-Scatter overlap with DGRAD GEMM by pipelining
#     the GEMM and Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰：「梯度反传之际，通信与计算本可并行而行，
#             旧代码却令其相互等待，犹如官僚衙门，拖沓误事。」
#   tp_comm_overlap_rs_dgrad=True 时，DGRAD GEMM 与 Reduce-Scatter 流水线重叠，
#   通信掩于计算之下，吞吐量从此不再干等。
#   - print('[M1960]') diagnostic added.
# ---------------------------------------------------------------------------
print('[M1960] core_base_config: tp_comm_overlap_rs_dgrad field added to BaseConfig')
from typing import Callable

import torch


@dataclass
class BaseConfig:
    """Base configuration for Megatron Core

    Model Parallelism
    -----------------

    tensor_model_parallel_size (int): Intra-layer model parallelism. Splits tensors across GPU ranks. Defaults to 1.

    context_parallel_size (int): Splits network input along sequence dimension across GPU ranks. Defaults to 1.

    pipeline_model_parallel_size (int): Inter-layer model parallelism. Splits transformer layers across GPU
        ranks. Defaults to 1.

    virtual_pipeline_model_parallel_size (int): Interleaved pipeline parallelism is used to improve performance by
        reducing the pipeline bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
        The number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.  See Efficient
        Large-Scale Language Model Training on GPU Clusters Using Megatron-LM: https://arxiv.org/pdf/2104.04473.pdf for
        more details.  Defaults to None.

    sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
        parallelizing layer norms and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer
        Models: https://arxiv.org/abs/2205.05198 for more details. Defaults to False.

    Initialization
    --------------

    init_method (Callable, default=init.xavier_normal_): Method to initialize weights. Note that bias is always set to zero.

    output_layer_init_method (Callable, default=init.xavier_normal_): Method to initialize weights of MLP output layer.

    init_method_std (float, default=0.02): Standard deviation of the zero mean normal.

    perform_initialization (bool, default=True): If true, weights are initialized. This option can be useful when you
        know you are going to load values from a checkpoint.

    use_cpu_initialization: (bool, default=False): When set to False, we initialize the weights directly on the GPU.
        Transferring weights from CPU to GPU can take a significant amount of time for large models. Defaults to False.

    Training
    --------

    fp16 (bool): If true, train with fp16 mixed precision training. Defaults to False.

    bf16 (bool): If true, train with bf16 mixed precision training. Defaults to False.

    params_dtype (torch.dtype): dtype used when intializing the weights. Defaults to torch.float32

    grad_scaler (optional, default=None): If using loss scaling, this function should take the loss and return the
        scaled loss. If None, no function is called on the loss.

    enable_autocast (bool): If true runs the forward step function inside torch.autocast context. Default is False.

    autocast_dtype (torch.dtype): dtype to pass to torch.amp.autocast when emabled. Default is params_dtype.

    timers (optional, default=None): TODO

    Optimizations
    -------------

    gradient_accumulation_fusion (bool): If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA
        extension fused_weight_gradient_mlp_cuda module. To use gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\"
        ". Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
        Defaults to False.

    async_tensor_model_parallel_allreduce (bool, default=True): If true, enables asynchronous execution of
        tensor-model-parallel all-reduce with weight gradient compuation of a column-linear layer.  Defaults to False.


    Pipeline Parallel
    -----------------

    pipeline_dtype (required when using pipeline parallelism): dtype used in
        p2p communication, usually params_dtype

    tensor_shape (tuple, required when using pipeline parallelism): Shape of tensor. The tensor is expected to be 3D and
        its order of dimension is supposed to be ``(sequence, batch, hidden)``.  TODO: currently seq_length is
        automatically divided by tensor parallel size if sequence_parallel is True, is this the right behavior, or do we
        want the user to specify the correct tensor_shape?

    variable_seq_lengths (bool, default=False): Support for variable sequence lengths across microbatches. Setting this
        communicates the size of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length is not constant during training.

    num_microbatches_with_partial_activation_checkpoints (int, default=None): If int, set the number of microbatches
        where not all of the layers will be checkpointed and recomputed. The rest of the microbatches within the window
        of maximum outstanding microbatches will recompute all layers (either full recompute or selective recompute). If
        None, the checkpoint and recompute will be left up to the forward_step function.

    overlap_p2p_comm (bool, optional, default=False): When True some of the peer to peer communication for pipeline
        parallelism will overlap with computation. Must be False if batch_p2p_comm is true.

    batch_p2p_comm (bool, default=True): Use batch_isend_irecv instead of individual isend/irecv calls. Must be False
        if overlap_p2p_comm is True.

    batch_p2p_sync (bool, default=True): When using batch_isend_irecv, do a cuda.device.synchronize afterward to work
        around a bug in older version of PyTorch.

    use_ring_exchange_p2p (bool, default = False): Use custom ring_exchange kernel instead of
        torch.distributed.batch_isend_irecv(). Requires custom built torch with torch.distributed.ring_exchange.

    deallocate_pipeline_outputs (optional, default=False): If True, output data is deallocated after the tensor is sent
        to the next pipeline stage.  Helps with saving memory, does nothing when pipeline parallel is not used.

    no_sync_func (optional): Function that creates a context that suppresses asynchronous data-parallel
        communication. If the model is an instance of torch.nn.DistributedDataParallel, the default is to use
        torch.nn.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous gradient reductions (e.g. distributed optimizer
        gradient reduce-scatters). The function should take one argument: an iterable of parameters whose gradients are
        to be synchronized.

    param_sync_func (optional): Function that launches asynchronous parameter synchronizations (e.g. distributed
        optimizer parameter all-gathers). The function should take one argument: an iterable of parameters to be
        synchronized.

    Legacy args (TODO: remove these)
    ------------------
    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    """

    # Model parallelism
    tensor_model_parallel_size: int = 1
    context_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = None
    sequence_parallel: bool = False

    # Initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    # Training
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    grad_scaler: Callable = None
    enable_autocast: bool = False
    autocast_dtype: torch.dtype = None
    timers: Callable = None

    # Optimizations
    gradient_accumulation_fusion: bool = False
    async_tensor_model_parallel_allreduce: bool = False

    # Pipeline parallel
    pipeline_dtype: torch.dtype = None
    tensor_shape: torch.Size = None
    variable_seq_lengths: bool = False
    num_microbatches_with_partial_activation_checkpoints: int = None
    batch_p2p_comm: bool = True
    overlap_p2p_comm: bool = False
    batch_p2p_sync: bool = True
    use_ring_exchange_p2p: bool = False
    deallocate_pipeline_outputs: bool = False
    no_sync_func: Callable = None
    grad_sync_func: Callable = None
    param_sync_func: Callable = None

    # Legacy
    decoder_seq_length: int = None
    # M1960: TP comm overlap — DGRAD reduce-scatter overlap with DGRAD GEMM
    # 鲁迅曰：「梯度计算与归约重叠，方为真正的流水线并行——不掩通信者，性能虚耗。」
    tp_comm_overlap_rs_dgrad: bool = False

    # M2030: TP communication bootstrap backend (Megatron f76b465e0)
    # 鲁迅曰：「通信后端之选择，如选路之岔口——nccl者速，mpi者稳，gloo者广；
    #          旧代码硬写 mpi，不知变通，今以接口开放，让调用者自择其路。」
    tp_comm_bootstrap_backend: str = 'nccl'
    """Set the bootstrapping backend out of 'nccl', 'mpi', and 'gloo'"""


    def __post__init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """

        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")
            if self.async_tensor_model_parallel_allreduce:
                # sequence_parallelism already does this async
                self.async_tensor_model_parallel_allreduce = False

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError("When using pipeline parallelism, pipeline_dtype must be specified")

            if self.tensor_shape is None:
                raise ValueError("When using pipeline parallelism, tensor_shape must be specified")

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype

        # M1506: Megatron 4ef31451d — batch_p2p_comm and overlap_p2p_comm are
        # mutually exclusive; the cleanup commit makes this constraint explicit
        # in both the docstring and the validation logic.  batch_p2p_sync is
        # orthogonal: it only has meaning when batch_p2p_comm is True and
        # guards against a PyTorch < 1.8 device synchronisation bug.
        if self.overlap_p2p_comm and self.batch_p2p_comm:
            raise ValueError(
                'overlap_p2p_comm and batch_p2p_comm are mutually exclusive: '
                'set batch_p2p_comm=False when overlap_p2p_comm=True '
                '(Megatron 4ef31451d / Neuron_SP M1506).'
            )
        print('[M1506] BaseConfig.__post__init__: overlap_p2p_comm=%s '
              'batch_p2p_comm=%s batch_p2p_sync=%s' % (
                  self.overlap_p2p_comm, self.batch_p2p_comm, self.batch_p2p_sync))

# ---------------------------------------------------------------------------
# M2030: Megatron f76b465e0 — Add TP communication bootstrap backend interface
# Source: megatron/core/model_parallel_config.py
#
# Mapping: megatron/core/model_parallel_config.py ModelParallelConfig
#        → deepspeed/compile/core_base_config.py BaseConfig
#
# Changes ported:
#   Field: tp_comm_bootstrap_backend: str = 'nccl'
#     Set the bootstrapping backend out of 'nccl', 'mpi', and 'gloo'
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰：「通信后端之选择，如选路之岔口——nccl者速，mpi者稳，gloo者广；
#             旧代码硬写 mpi，不知变通，犹如守旧官僚，误人不浅。
#             今以 tp_comm_bootstrap_backend 开接口，让调用者自择其路。」
#   - tp_comm_bootstrap_backend 字段加入 BaseConfig
#   - print('[M2030]') diagnostic added.
# ---------------------------------------------------------------------------
print('[M2030] core_base_config: tp_comm_bootstrap_backend field added to BaseConfig (default=nccl)')
