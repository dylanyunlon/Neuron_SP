# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Ported from Megatron-LM commit 98550bf32ab32e3bddeec29ccaa21b91080bf8a8
# M1424: Add PipelineConfig.

print('[M1424]')

# ---------------------------------------------------------------------------
# M1506: Megatron 4ef31451d — Fixes/cleanup from overlap p2p merge
# Source: megatron/core/model_parallel_config.py (NVIDIA/Megatron-LM commit 4ef31451d)
#
# Mapping: megatron/core/model_parallel_config.py ModelParallelConfig
#        → deepspeed/runtime/pipe/pipeline_config.py PipelineConfig
#
# Changes ported (mirror of core_base_config.py / BaseConfig):
#   1. Docstring updated with overlap_p2p_comm, batch_p2p_comm (default True),
#      batch_p2p_sync entries replacing the old batch_p2p_comm=False entry.
#   2. Fields:
#        overlap_p2p_comm: bool = False   (NEW)
#        batch_p2p_comm: bool = True      (was False)
#        batch_p2p_sync: bool = True      (NEW)
#   3. __post__init__: mutual-exclusion guard added.
#
# 20% adaptation: ValueError + print diagnostic in __post__init__.
# ---------------------------------------------------------------------------

print('[M1506]')
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# M1860: Megatron c2df7e3c1 — Only call finalize_model_grads when available
# Source: megatron/core/model_parallel_config.py (NVIDIA/Megatron-LM commit c2df7e3c1)
#
# Mapping: megatron/core/model_parallel_config.py ModelParallelConfig
#        → deepspeed/runtime/pipe/pipeline_config.py PipelineConfig
#
# Changes ported:
#   1. Docstring: added Parallelism section documenting finalize_model_grads_func.
#   2. Field: finalize_model_grads_func: Callable = None
#
# 20% adaptation: field added alongside param_sync_func (parallelism group);
# print diagnostic in __post__init__ reports the func presence.
# ---------------------------------------------------------------------------

print('[M1860]')
@dataclass
class PipelineConfig:
    """Pipeline configuration for Megatron Core

    sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
        parallelizing layer norms and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer
        Models: https://arxiv.org/abs/2205.05198 for more details. Defaults to False.
    
    pipeline_dtype (required): dtype used in p2p communication, usually params_dtype

    grad_scaler (optional, default=None): If using loss scaling, this function should take the loss and return the
        scaled loss. If None, no function is called on the loss.

    enable_autocast (bool): If true runs the forward step function inside torch.autocast context. Default is False.

    autocast_dtype (torch.dtype): dtype to pass to torch.amp.autocast when emabled. Default is pipeline_dtype.

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
    
    timers (optional, default=None): TODO

    finalize_model_grads_func (optional): Function that finalizes gradients on all workers.
        Could include ensuring that grads are all-reduced across data parallelism, pipeline
        parallelism, and sequence parallelism dimensions.  When None, finalization is skipped.

        Legacy args (TODO: remove these)
    ------------------
    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    """

    sequence_parallel: bool = False
    grad_scaler: Callable = None
    enable_autocast: bool = False
    autocast_dtype: torch.dtype = None
    timers: Callable = None

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
    # Parallelism (M1860)
    finalize_model_grads_func: Callable = None

    # Legacy
    decoder_seq_length: int = None

    def __post__init__(self):
        if self.pipeline_dtype is None:
            raise ValueError("When using pipeline parallelism, pipeline_dtype must be specified")

        if self.tensor_shape is None:
            raise ValueError("tensor_shape must be provided")
        
        if self.autocast_dtype is None:
            self.autocast_dtype = self.pipeline_dtype

        if self.decoder_seq_length is None:
            self.decoder_seq_length = self.tensor_shape[0]

        # M1506: Megatron 4ef31451d — enforce mutual exclusion between
        # overlap_p2p_comm and batch_p2p_comm (matching BaseConfig validation).
        if self.overlap_p2p_comm and self.batch_p2p_comm:
            raise ValueError(
                'overlap_p2p_comm and batch_p2p_comm are mutually exclusive '
                '(Megatron 4ef31451d / Neuron_SP M1506).'
            )
        print('[M1860] PipelineConfig.__post__init__: finalize_model_grads_func=%s' % (
            self.finalize_model_grads_func is not None))
        print('[M1506] PipelineConfig.__post__init__: overlap_p2p_comm=%s '
              'batch_p2p_comm=%s batch_p2p_sync=%s' % (
                  self.overlap_p2p_comm, self.batch_p2p_comm, self.batch_p2p_sync))
