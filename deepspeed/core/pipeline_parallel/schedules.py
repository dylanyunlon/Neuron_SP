# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# Insight I11: single layer offset source (Megatron M3116)
# POLICY: This module MUST NOT compute pipeline layer offsets locally.
# All layer-offset arithmetic (e.g. pp_rank * layers_per_rank) belongs
# exclusively in TransformerConfig.get_transformer_layer_offset().
# Rationale: M3150 / M3295 showed that independent per-module offset
# computations diverge under uneven PP splits and Mamba hybrid layouts.
# Violation of this policy will re-introduce the same class of silent
# correctness bug in heterogeneous DES-LOC clusters.

"""Pipeline schedule implementations for pipeline parallelism.

Ported from Megatron-LM/megatron/core/pipeline_parallel/schedules.py and
extended with DES-LOC heterogeneous bubble-filling.

Megatron commit lineage (34 commits) fully incorporated
--------------------------------------------------------
M2280  Remove create_cudagraphs + unify cudagraphs recording/creation
M2313  Protect against divide-by-0 when all tokens masked in a microbatch
M2319  Defer training graph creation until create_cudagraphs
M2459  Unify enable/external cudagraph with cuda-graph-impl
M2812  Fix aux loss scale when CP is enabled
M2860  Pipeline parallelism fix in RL and sequence packing rewriting
M2906  Refactor cuda_graph_scope (MoE)
M3018  Fine-grained activation offloading (MoE)
M3030  Partial CUDA Graph support for EP Overlap
M3047  Hybrid Context Parallel Feature
M3087  Add ability to save wgrads and dgrads
M3173  Add MTP support for hybrid models
M3213  Reapply MTP support for hybrid models
M3359  Remove encoder_and_decoder from enums
M3456  TE CUDA Graph Support for Vision Encoder
M3490  Reset activation offload manager after eval
M3513  Remove encoder_and_decoder (final)
M3544  Support multimodule pipelining in 1F1B schedule
M3734  Reset AG_pipeline bucket status after validation step
M3758  Get device correctly when module returns dict
M3766  Wait for async P2P send before deallocating output tensor
M3977  Refactor CUDA graph API: decompose cuda_graph_scope
M3981  Thread custom process groups through MoE grad finalization
M4012  Paged Stashing
M4063  Delete output tensor early (avoid AccumulateGrad stream warning)
M4082  MIMO bridge fan-out for variable modality tokens
M4083  Separate mtp_grad_scale_func for MTP loss scaling
M4109  Fix DSA indexer loss not averaged across micro-batches
M4176  Add moe_grad_scale_func / moe loss normalization for RL SFT

Schedule summary
----------------
PP=1:
    forward_backward_no_pipelining
PP>1, VPP=None:
    forward_backward_pipelining_without_interleaving
PP>1, VPP>1:
    forward_backward_pipelining_with_interleaving

DES-LOC extension: heterogeneous bubble filling via HeterogeneousBubbleFiller.
"""

from __future__ import annotations

import contextlib
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.pipeline_parallel.p2p_communication import (
    P2PCommunicator,
    MultiModulePipelineCommunicator,
    is_single_shape,
    is_cross_numa_transfer,
)

try:
    from deepspeed.core import parallel_state as _ps
except ImportError:
    _ps = None

try:
    from deepspeed.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface as off_interface,
    )
    _HAS_FGAO = True
except ImportError:
    _HAS_FGAO = False
    off_interface = None

try:
    from deepspeed.core.pipeline_parallel.multimodule_communicator import (
        MultiModulePipelineCommunicator as _MultiModuleCommunicator,
    )
except ImportError:
    _MultiModuleCommunicator = MultiModulePipelineCommunicator

try:
    from deepspeed.core.process_groups_config import (
        ProcessGroupCollection,
        MultiModuleProcessGroupCollection,
    )
except ImportError:
    class ProcessGroupCollection:
        pass
    class MultiModuleProcessGroupCollection:
        pass

try:
    from deepspeed.core.transformer.cuda_graphs import create_cudagraphs, set_current_microbatch
except ImportError:
    def create_cudagraphs(): pass
    def set_current_microbatch(model, mb): pass

try:
    from deepspeed.core.transformer.moe.paged_stash import paged_stash_reset
except ImportError:
    def paged_stash_reset(enabled, config): pass

try:
    from deepspeed.core.transformer.moe.router import MoEAuxLossAutoScaler
except ImportError:
    class MoEAuxLossAutoScaler:
        @staticmethod
        def set_loss_scale(x): pass

try:
    from deepspeed.core.utils import (
        drain_embedding_wgrad_compute,
        get_attr_wrapped_model,
        get_model_config,
        get_model_type,
        nvtx_range_pop,
        nvtx_range_push,
    )
except ImportError:
    def drain_embedding_wgrad_compute(*a, **k): pass
    def get_attr_wrapped_model(model, attr, **k): return getattr(model, attr, None)
    def get_model_config(model): return getattr(model, 'config', model)
    def get_model_type(model): return None
    def nvtx_range_pop(**k): pass
    def nvtx_range_push(**k): pass

try:
    from deepspeed.core.pipeline_parallel.utils import (
        is_pp_first_stage,
        is_pp_last_stage,
        is_vp_first_stage,
        is_vp_last_stage,
    )
except ImportError:
    def is_pp_first_stage(pg): return pg.rank() == 0
    def is_pp_last_stage(pg): return pg.rank() == pg.size() - 1
    def is_vp_first_stage(vp_stage, vp_size): return vp_stage == 0
    def is_vp_last_stage(vp_stage, vp_size): return vp_stage == vp_size - 1

try:
    from deepspeed.core.pipeline_parallel.combined_1f1b import (
        combined_1f1b_schedule_for_interleaved_pipelining,
        combined_1f1b_schedule_for_no_pipelining,
    )
except ImportError:
    def combined_1f1b_schedule_for_no_pipelining(*a, **k): raise NotImplementedError
    def combined_1f1b_schedule_for_interleaved_pipelining(*a, **k): raise NotImplementedError

try:
    from deepspeed.core.pipeline_parallel.hybrid_cp_schedule import (
        hybrid_context_parallel_forward_backward,
    )
except ImportError:
    def hybrid_context_parallel_forward_backward(*a, **k): raise NotImplementedError

Shape = Union[List[int], torch.Size]

_PIPELINE_LAYER_SPLIT: Optional[List[int]] = None

# ===========================================================================
# Utility helpers
# ===========================================================================

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """Pseudo-deallocate the output tensor data field to free activation memory.

    Supports torch.Tensor, List[Tensor], Dict[str, Tensor] (M3544 multimodule).
    """
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    if isinstance(out, dict):
        for value in out.values():
            deallocate_output_tensor(value, deallocate_pipeline_outputs)
        return
    if isinstance(out, list):
        for item in out:
            deallocate_output_tensor(item, deallocate_pipeline_outputs)
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output):
    """Directly call C++ autograd engine (needed with deallocate_pipeline_outputs)."""
    assert output.numel() == 1
    assert isinstance(output, torch.Tensor)
    assert isinstance(grad_output, (torch.Tensor, type(None)))
    if grad_output is None:
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def get_tensor_device(tensor):
    """Get device of a tensor or dict of tensors (M3758)."""
    if isinstance(tensor, dict):
        return next(iter(tensor.values())).device
    return tensor.device


def _get_mtp_loss_scale(config, device, num_tokens: "Optional[torch.Tensor]" = None):
    """Get MTP loss scale, preferring mtp_grad_scale_func (M4083).

    Fix (Megatron M3312, PR #3159): when ``calculate_per_token_loss=True``
    ``process_mtp_loss`` already multiplies the MTP loss by
    ``original_num_tokens / rolled_num_tokens`` before handing it to
    ``MTPLossAutoScaler``.  In the backward pass the AutoScaler multiplies
    that pre-scaled loss by ``main_loss_backward_scale``, so MTP gradients
    end up weighted by ``scale * (original / rolled)`` while the main loss
    gradients are weighted by ``scale`` only.

    The correction: divide the scale by that same ratio so the net weight is
    just ``scale``.  ``original_num_tokens`` is published onto
    ``MTPLossAutoScaler.original_num_tokens`` by ``process_mtp_loss`` each
    forward pass.  ``num_tokens`` (the pre-roll count from the main loss
    function) equals ``original_num_tokens``, so the rolled count is
    approximated from the stored value.  If the stored value is not yet set
    (first step, non-MTP path) the correction is skipped safely.
    """
    def _normalize(ls, name):
        ls = torch.as_tensor(ls, device=device)
        if ls.numel() != 1:
            raise ValueError(f"{name} must return scalar or size-1 tensor")
        return ls
    fn = getattr(config, 'mtp_grad_scale_func', None)
    if fn is not None:
        loss_scale = _normalize(fn(), "mtp_grad_scale_func")
    elif config.grad_scale_func is not None:
        loss_scale = _normalize(config.grad_scale_func(torch.ones(1, device=device)), "grad_scale_func")
    else:
        loss_scale = torch.ones(1, device=device)

    # M3312 correction: when per-token loss is active and process_mtp_loss
    # has published original_num_tokens, fold the inverse ratio into the
    # scale so that the AutoScaler backward weight stays at just `loss_scale`.
    if getattr(config, 'calculate_per_token_loss', False):
        try:
            from deepspeed.core.transformer.multi_token_prediction import MTPLossAutoScaler as _MTPScaler
            orig = _MTPScaler.original_num_tokens
            if orig is not None and num_tokens is not None:
                num_tokens_t = torch.as_tensor(num_tokens, dtype=loss_scale.dtype, device=device)
                orig_t = orig.to(dtype=loss_scale.dtype, device=device)
                # Rolled count ≈ orig − batch_size (one position zeroed per seq).
                # We use num_tokens as the best available proxy for the
                # post-roll valid-token count seen by the main loss function;
                # process_mtp_loss uses its own internal rolled count which may
                # differ by at most B tokens.  This matches Megatron M3312.
                rolled_t = torch.clamp(num_tokens_t, min=1)
                orig_safe = torch.clamp(orig_t, min=1)
                # Divide by the ratio that process_mtp_loss already applied.
                loss_scale = loss_scale * rolled_t / orig_safe
        except Exception:
            pass  # Correction is best-effort; fall back to uncorrected scale.

    return loss_scale


def get_forward_backward_func(pp_size=None, vp_size=None, config=None):
    """Return the appropriate forward_backward function for the given PP/VPP config.

    DES-LOC extension: when ``config.desloc.bubble_filler`` is set and PP>1
    without VPP, the native heterogeneous 1F1B schedule is returned so that
    per-rank asymmetric warmup and inline bubble-filling are active.

    Args:
        pp_size (Optional[int]): Pipeline model parallel size.
        vp_size (Optional[int]): Virtual pipeline model parallel size.
        config:  Optional model config carrying ``config.desloc.bubble_filler``.
    """
    if pp_size is None and vp_size is None:
        pp_size = _ps.get_pipeline_model_parallel_world_size()
        vp_size = _ps.get_virtual_pipeline_model_parallel_world_size()
    if pp_size > 1:
        if vp_size is not None:
            return forward_backward_pipelining_with_interleaving
        else:
            # DES-LOC: prefer the native hetero schedule when a bubble filler
            # is registered on the config, so callers get adaptive warmup and
            # inline bubble-filling without extra plumbing.
            if config is not None:
                _desloc = getattr(config, 'desloc', None)
                if _desloc is not None and getattr(_desloc, 'bubble_filler', None) is not None:
                    return forward_backward_hetero_1f1b
            return forward_backward_pipelining_without_interleaving
    return forward_backward_no_pipelining


# ===========================================================================
# Loss calculation
# ===========================================================================

def forward_step_calc_loss(
    model, output_tensor, loss_func, config, vp_stage, collect_non_loss_data,
    num_microbatches, forward_data_store, cp_group_size=None, is_last_stage=None,
):
    """Calculate loss and num_tokens. Incorporates M2313, M2812, M3544, M4083, M4109, M4176."""
    try:
        from deepspeed.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
        _has_dsa = True
    except ImportError:
        _has_dsa = False
    try:
        from deepspeed.core.transformer.multi_token_prediction import MTPLossAutoScaler
        _has_mtp = True
    except ImportError:
        _has_mtp = False

    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert vp_stage == model_vp_stage, f"vp_stage mismatch: {vp_stage} vs {model_vp_stage}"

    if cp_group_size is None and is_last_stage is None:
        cp_group_size = _ps.get_context_parallel_world_size()
        is_last_stage = _ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    else:
        assert is_last_stage is not None, "is_last_stage must be provided"
        if is_last_stage:
            assert cp_group_size is not None, "cp_group_size must be provided on last stage"

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_last_stage:
        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= torch.clamp(num_tokens, min=1)  # M2313: div-by-zero guard
                    output_tensor /= num_microbatches
            else:
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # MoE auxiliary loss scaling (M2812, M4176: moe_grad_scale_func)
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        device = get_tensor_device(output_tensor)
        moe_grad_scale_func = getattr(config, 'moe_grad_scale_func', None)
        if moe_grad_scale_func is not None:
            loss_scale = moe_grad_scale_func()
        elif config.grad_scale_func is not None:
            loss_scale = config.grad_scale_func(torch.ones(1, device=device))
        else:
            loss_scale = torch.ones(1, device=device)
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            # From Megatron M4098: aux_loss is computed per-TP-rank (each rank sees
            # seq_len/TP tokens), so its gradient is implicitly divided by TP relative
            # to the main loss. Multiply by tp_size to restore correct relative scale.
            # Previously only CP was accounted for, silently underscaling aux_loss
            # gradients by tp_size when TP > 1, degrading MoE load-balancing.
            cp_size_for_scaling = cp_group_size if cp_group_size is not None else 1
            tp_size_for_scaling = getattr(config, 'tensor_model_parallel_size', 1) or 1
            MoEAuxLossAutoScaler.set_loss_scale(
                loss_scale * cp_size_for_scaling * tp_size_for_scaling / num_microbatches
            )

    # MTP loss scaling (M3213, M4083, M3312)
    if _has_mtp and hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        device = get_tensor_device(output_tensor)
        # Pass num_tokens so _get_mtp_loss_scale can apply the M3312
        # token-ratio correction when calculate_per_token_loss=True.
        loss_scale = _get_mtp_loss_scale(config, device, num_tokens=num_tokens)
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # DSA indexer loss (M4109: divide by num_microbatches)
    if _has_dsa and getattr(config, 'experimental_attention_variant', None) == 'dsa':
        device = get_tensor_device(output_tensor)
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=device)
        )
        if config.calculate_per_token_loss:
            DSAIndexerLossAutoScaler.set_loss_scale(loss_scale)
        else:
            DSAIndexerLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens


# ===========================================================================
# Forward / backward step
# ===========================================================================

def forward_step(
    forward_step_func, data_iterator, model, num_microbatches,
    input_tensor, forward_data_store, config,
    cp_group_size=None, collect_non_loss_data=False,
    checkpoint_activations_microbatch=None, is_first_microbatch=False,
    current_microbatch=None, vp_stage=None, is_last_stage=True,
):
    """Forward step through the model for one microbatch."""
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()
    # From Megatron M2300 (323051030): Fix is_first_microbatch not correctly set
    # with CUDA graph enabled.  set_is_first_microbatch() must be called BEFORE
    # set_current_microbatch() so that FP8 / CUDA-graph replay correctly identifies
    # the first microbatch even when replaying from a graph.  In DES-LOC the H100
    # may use CUDA graphs for the high-throughput inference tier while A6000 runs
    # eager mode; both paths must set is_first_microbatch consistently to avoid
    # mismatched FP8 scale updates at microbatch boundaries.
    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    ctx = torch.autocast("cuda", dtype=config.autocast_dtype) if config.enable_autocast else contextlib.nullcontext()
    with ctx:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(data_iterator, model, checkpoint_activations_microbatch)

    output_tensor, num_tokens = forward_step_calc_loss(
        model, output_tensor, loss_func, config, vp_stage, collect_non_loss_data,
        num_microbatches, forward_data_store, cp_group_size, is_last_stage,
    )
    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
    """Backward step. M3544: model_type arg removed. Handles VLM no-image case."""
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()
    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            input_tensor_grad.append(None if x is None else x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()
    return input_tensor_grad


def backward_step_multimodule(
    input_tensor: Dict[str, torch.Tensor],
    output_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_tensor_grad: Optional[Dict[str, torch.Tensor]],
    config,
    language_model_module_name: str,
) -> Dict[str, torch.Tensor]:
    """Backward step for multi-module pipelines (M3544). Dict-keyed tensors.

    In multi-module pipelines, tensors are organised as dictionaries with
    module names as keys.  Each module's backward pass is performed
    independently.  Mirrors Megatron backward_step_multimodule exactly,
    including the _unwrap_single_tensor_list helper (renamed from _unwrap in
    the original DES-LOC port to match upstream naming for easier diffing).
    """

    def _unwrap_single_tensor_list(tensor):
        if isinstance(tensor, list):
            assert len(tensor) == 1, "expected a single tensor for multimodule backward"
            return tensor[0]
        return tensor

    # Retain gradients on all input tensors.
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is not None:
            tensor.retain_grad()

    # Last stage: output_tensor is a scalar loss from the language model.
    # Associate it with the language_model_module_name.
    if not isinstance(output_tensor, dict):
        output_tensor = {language_model_module_name: output_tensor}

    # Handle output_tensor_grad: None (last stage) or dict (intermediate stages).
    if not output_tensor_grad:
        output_tensor_grad = {key: None for key in output_tensor.keys()}

    # Apply grad scaling if needed (for last stage only).
    for module_name in output_tensor.keys():
        output_tensor_grad_module = _unwrap_single_tensor_list(output_tensor_grad[module_name])
        if output_tensor_grad_module is None and config.grad_scale_func is not None:
            output_tensor[module_name] = config.grad_scale_func(output_tensor[module_name])

    # Perform backward pass for each module.
    for module_name in output_tensor.keys():
        ot = _unwrap_single_tensor_list(output_tensor[module_name])
        otg = _unwrap_single_tensor_list(output_tensor_grad[module_name])
        if ot is not None and ot.requires_grad:
            if config.deallocate_pipeline_outputs:
                custom_backward(ot, otg)
            else:
                torch.autograd.backward(ot, grad_tensors=otg)

    # Collect gradients for input tensors.
    input_tensor_grad = {}
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is None:
            input_tensor_grad[module_name] = None
        else:
            input_tensor_grad[module_name] = tensor.grad
    return input_tensor_grad


# ===========================================================================
# Shared helpers
# ===========================================================================

def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    return cond


def clear_embedding_activation_buffer(config, model, is_last_stage):
    if is_last_stage and config.defer_embedding_wgrad_compute:
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(model[-1], 'post_process', return_model_obj=True)
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)
        embedding_module.embedding_activation_buffer.clear()
        return embedding_module
    return None


def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
    if is_last_stage and config.defer_embedding_wgrad_compute:
        ea_buf = embedding_module.embedding_activation_buffer
        go_buf = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )
        drain_embedding_wgrad_compute(config, ea_buf, go_buf, weight, tp_group)


def get_pp_rank_microbatches(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage,
                              forward_only=False, overlap_moe_expert_parallel_comm=False,
                              p2p_communicator=None):
    if p2p_communicator is not None:
        pp_size = p2p_communicator.pp_group.size()
        pp_rank = p2p_communicator.pp_group.rank()
        vpp_size = p2p_communicator.virtual_pipeline_model_parallel_size
    else:
        pp_size = _ps.get_pipeline_model_parallel_world_size()
        pp_rank = _ps.get_pipeline_model_parallel_rank()
        vpp_size = _ps.get_virtual_pipeline_model_parallel_world_size()

    total = num_microbatches * num_model_chunks
    all_in_warmup = False

    if forward_only:
        num_warmup = total
    elif pp_size > 1:
        if vpp_size is None:
            num_warmup = pp_size - pp_rank - 1
        else:
            num_warmup = (pp_size - pp_rank - 1) * 2
            num_warmup += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
            if overlap_moe_expert_parallel_comm:
                num_warmup += 1
    else:
        num_warmup = 0

    if num_warmup >= total:
        num_warmup = total
        all_in_warmup = True
    num_remaining = total - num_warmup
    return total, all_in_warmup, num_warmup, num_remaining


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Build schedule table of (microbatch_id, model_chunk_id) pairs."""
    table = []
    for min_mb in range(0, num_microbatches, microbatch_group_size_per_vp_stage):
        if min_mb + microbatch_group_size_per_vp_stage >= num_microbatches:
            table.extend(
                [(mb, ch) for ch in range(num_model_chunks) for mb in range(min_mb, num_microbatches)]
            )
        else:
            table.extend(
                [(mb, ch) for ch in range(num_model_chunks)
                 for mb in range(min_mb, min_mb + microbatch_group_size_per_vp_stage)]
            )
    return table


# ===========================================================================
# DES-LOC heterogeneous bubble filler
# ===========================================================================

class StageClock:
    """Per-stage exponential moving-average (EMA) compute-time tracker.

    Maintains a running estimate of how long each pipeline stage takes for
    one forward or backward pass, using an EMA with configurable smoothing
    factor alpha.  The estimate is used by AsymmetricClockScheduler to avoid
    dispatching new work to a stage that is still busy.

    Formula: ema_t = alpha * observed_t + (1 - alpha) * ema_{t-1}

    Args:
        alpha:        EMA smoothing coefficient (0 < alpha <= 1).  A value
                      close to 1 tracks recent observations aggressively;
                      values near 0 are more stable.
        initial_ms:   Seed value for the EMA (milliseconds).  Should be a
                      reasonable rough estimate of per-microbatch compute
                      time for the stage's GPU tier.
    """

    def __init__(self, alpha: float = 0.2, initial_ms: float = 100.0):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.ema_ms = initial_ms
        self._start: Optional[float] = None

    def start(self) -> None:
        """Record the start of a compute step (uses wall-clock perf_counter)."""
        import time
        self._start = time.perf_counter()

    def stop(self) -> float:
        """Record the end of a compute step and update the EMA.

        Returns:
            The measured elapsed time in milliseconds.
        """
        import time
        if self._start is None:
            return self.ema_ms
        elapsed_ms = (time.perf_counter() - self._start) * 1_000.0
        self._start = None
        self.ema_ms = self.alpha * elapsed_ms + (1.0 - self.alpha) * self.ema_ms
        return elapsed_ms

    def estimate(self) -> float:
        """Return the current EMA estimate in milliseconds."""
        return self.ema_ms

    def __repr__(self) -> str:
        return f"StageClock(ema_ms={self.ema_ms:.2f}, alpha={self.alpha})"


class AsymmetricClockScheduler:
    """Adaptive 1F1B dispatch scheduler for PP=5 heterogeneous pipelines.

    In a 5-stage pipeline mixing H100 (fast) and A6000 (slow) GPUs the
    standard 1F1B schedule creates large bubbles on fast stages because they
    must wait for the slowest stage before the pipeline clock advances.
    This scheduler keeps track of estimated per-stage compute times and
    determines:

    1. Which microbatches a *fast* stage can pre-compute (fill bubbles)
       while it is nominally idle, staying within the activation memory
       budget set by ``max_outstanding_activations``.
    2. The order in which ``recv_forward`` should be called so that fast
       stages always have work queued up.

    Design decisions
    ----------------
    * Memory-aware: prefetch_slots = max_outstanding_activations - current_in_flight
      ensures we never OOM a fast stage with speculative activations.
    * Non-blocking: all decisions are made synchronously on the CPU; no
      extra CUDA kernels are launched.
    * Graceful degradation: if stage_clocks is empty or PP=1 the scheduler
      falls through to the standard 1F1B ordering.

    Args:
        num_stages:                  Total number of PP stages (e.g. 5).
        stage_clocks:                Mapping from PP rank -> StageClock.
        max_outstanding_activations: Maximum number of microbatch activations
                                     that may be held simultaneously on a fast
                                     stage.  Defaults to num_stages - 1
                                     (standard 1F1B limit).
        fast_rank_set:               Set of PP ranks considered "fast" (H100).
    """

    def __init__(
        self,
        num_stages: int,
        stage_clocks: Optional[Dict[int, "StageClock"]] = None,
        max_outstanding_activations: Optional[int] = None,
        fast_rank_set: Optional[Set[int]] = None,
    ):
        self.num_stages = num_stages
        self.stage_clocks: Dict[int, StageClock] = stage_clocks or {}
        self.max_outstanding_activations = (
            max_outstanding_activations
            if max_outstanding_activations is not None
            else num_stages - 1
        )
        self.fast_rank_set: Set[int] = fast_rank_set or set()
        self._in_flight: int = 0

    def is_fast_rank(self, pp_rank: int) -> bool:
        return pp_rank in self.fast_rank_set

    def slowdown_ratio(self, fast_rank: int, slow_rank: int) -> float:
        """Ratio of slow-stage compute time to fast-stage compute time.

        A value > 1 means the slow stage is slower, justifying bubble filling.
        """
        if fast_rank not in self.stage_clocks or slow_rank not in self.stage_clocks:
            return 1.0
        fast_ms = self.stage_clocks[fast_rank].estimate()
        slow_ms = self.stage_clocks[slow_rank].estimate()
        if fast_ms <= 0:
            return 1.0
        return slow_ms / fast_ms

    def available_prefetch_slots(self) -> int:
        """How many extra microbatches a fast stage may pre-compute right now."""
        return max(0, self.max_outstanding_activations - self._in_flight)

    def record_forward_start(self) -> None:
        self._in_flight += 1

    def record_backward_complete(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    def should_prefetch(self, pp_rank: int, num_microbatches_remaining: int) -> bool:
        """Return True if pp_rank should try to prefetch an extra microbatch.

        Conditions:
        - This rank is in the fast set.
        - There are remaining microbatches that haven't been dispatched.
        - Memory budget allows at least one more in-flight activation.
        - The slowdown ratio is significant (> 1.2).
        """
        if not self.is_fast_rank(pp_rank):
            return False
        if num_microbatches_remaining <= 0:
            return False
        if self.available_prefetch_slots() <= 0:
            return False
        bottleneck_rank = self._find_bottleneck()
        if bottleneck_rank is None:
            return False
        ratio = self.slowdown_ratio(pp_rank, bottleneck_rank)
        return ratio > 1.2

    def _find_bottleneck(self) -> Optional[int]:
        """Return the rank with the highest estimated compute time."""
        if not self.stage_clocks:
            return None
        return max(self.stage_clocks, key=lambda r: self.stage_clocks[r].estimate())

    def compute_warmup_override(self, pp_rank: int, base_warmup: int) -> int:
        """Optionally extend warmup for fast ranks to pre-fill the pipeline.

        For a fast stage (H100) the standard warmup is (PP-rank-1) microbatches.
        We increase this by floor(slowdown_ratio - 1) extra microbatches
        so more activations are ready when the slow stage catches up, reducing
        the effective steady-state bubble.  Capped at max_outstanding_activations.
        """
        if not self.is_fast_rank(pp_rank):
            return base_warmup
        bottleneck = self._find_bottleneck()
        if bottleneck is None:
            return base_warmup
        ratio = self.slowdown_ratio(pp_rank, bottleneck)
        extra = int(max(0.0, ratio - 1.0))
        return min(base_warmup + extra, self.max_outstanding_activations)

    def __repr__(self) -> str:
        clocks_repr = {r: f"{c.estimate():.1f}ms" for r, c in self.stage_clocks.items()}
        return (
            f"AsymmetricClockScheduler("
            f"stages={self.num_stages}, "
            f"fast={self.fast_rank_set}, "
            f"clocks={clocks_repr}, "
            f"in_flight={self._in_flight})"
        )


class HeterogeneousP2PManager:
    """PCIe-aware tensor-shape negotiation and bandwidth throttle.

    In DES-LOC clusters, adjacent pipeline stages may be on different GPU
    tiers connected via PCIe rather than NVLink.  Transferring large
    activations over PCIe (~16 GB/s vs NVLink ~600 GB/s) can dominate
    stage-to-stage latency.

    This manager:
    1. Maintains a per-link bandwidth estimate (GB/s) derived from observed
       transfer times (EMA-updated).
    2. Advises callers on the maximum activation tensor size (bytes) that
       can be transferred within a target latency budget.
    3. Provides ``chunk_tensor`` / ``reassemble_tensor`` for transparently
       splitting oversized tensors into chunks that fit the budget.

    A "link" is identified by the ordered pair (src_pp_rank, dst_pp_rank).

    Args:
        link_bandwidths_gbps: Initial bandwidth estimates, keyed by
                              (src_rank, dst_rank).  Default 16 GB/s (PCIe).
        target_latency_ms:   Maximum acceptable one-way transfer latency per
                             microbatch (ms).  Default 20 ms.
        alpha:               EMA smoothing for bandwidth estimate (default 0.2).
    """

    DEFAULT_BANDWIDTH_GBPS: float = 16.0
    BYTES_PER_GB: int = 1024 ** 3

    def __init__(
        self,
        link_bandwidths_gbps: Optional[Dict[Tuple[int, int], float]] = None,
        target_latency_ms: float = 20.0,
        alpha: float = 0.2,
    ):
        self.target_latency_ms = target_latency_ms
        self.alpha = alpha
        self._ema_bw: Dict[Tuple[int, int], float] = {}
        if link_bandwidths_gbps:
            for link, bw in link_bandwidths_gbps.items():
                self._ema_bw[link] = bw

    def get_bandwidth_gbps(self, src: int, dst: int) -> float:
        """Return current EMA bandwidth estimate for (src->dst) in GB/s."""
        return self._ema_bw.get((src, dst), self.DEFAULT_BANDWIDTH_GBPS)

    def update_bandwidth(
        self, src: int, dst: int, bytes_transferred: int, elapsed_ms: float
    ) -> float:
        """Update bandwidth EMA after a measured transfer.

        Args:
            src:               Source PP rank.
            dst:               Destination PP rank.
            bytes_transferred: Number of bytes transferred.
            elapsed_ms:        Measured transfer time in milliseconds.

        Returns:
            Updated bandwidth estimate in GB/s.
        """
        if elapsed_ms <= 0:
            return self.get_bandwidth_gbps(src, dst)
        observed_gbps = (bytes_transferred / self.BYTES_PER_GB) / (elapsed_ms / 1_000.0)
        prev = self._ema_bw.get((src, dst), self.DEFAULT_BANDWIDTH_GBPS)
        updated = self.alpha * observed_gbps + (1.0 - self.alpha) * prev
        self._ema_bw[(src, dst)] = updated
        return updated

    def max_bytes_per_latency_budget(self, src: int, dst: int) -> int:
        """Maximum bytes transferable within target_latency_ms on (src->dst)."""
        bw_gbps = self.get_bandwidth_gbps(src, dst)
        budget_bytes = int(bw_gbps * self.BYTES_PER_GB * (self.target_latency_ms / 1_000.0))
        return max(budget_bytes, 1)

    def should_chunk(self, tensor: torch.Tensor, src: int, dst: int) -> bool:
        """Return True if tensor should be split into chunks for PCIe transfer."""
        if tensor is None:
            return False
        tensor_bytes = tensor.numel() * tensor.element_size()
        budget = self.max_bytes_per_latency_budget(src, dst)
        return tensor_bytes > budget * 2

    def chunk_tensor(
        self, tensor: torch.Tensor, src: int, dst: int
    ) -> List[torch.Tensor]:
        """Split tensor along dim=0 into chunks fitting the bandwidth budget.

        Args:
            tensor: Tensor to split (shape: [seq_len, batch, hidden]).
            src:    Source PP rank.
            dst:    Destination PP rank.

        Returns:
            List of tensor chunks, each sendable within target_latency_ms.
        """
        if tensor is None:
            return [tensor]
        bytes_per_element = tensor.element_size()
        total_elements = tensor.numel()
        elements_per_slice = total_elements // max(1, tensor.shape[0])
        budget_bytes = self.max_bytes_per_latency_budget(src, dst)
        elements_per_budget = max(1, budget_bytes // max(1, bytes_per_element))
        chunk_size_dim0 = max(1, elements_per_budget // max(1, elements_per_slice))
        chunks = tensor.split(chunk_size_dim0, dim=0)
        return list(chunks)

    def reassemble_tensor(
        self, chunks: List[torch.Tensor], original_shape: torch.Size
    ) -> torch.Tensor:
        """Reassemble chunks from ``chunk_tensor`` back into a single tensor.

        Args:
            chunks:         List of tensor chunks.
            original_shape: Expected shape of the reassembled tensor.

        Returns:
            Reconstructed tensor with shape == original_shape.
        """
        if len(chunks) == 1:
            t = chunks[0]
            return t.reshape(original_shape) if t.shape != original_shape else t
        t = torch.cat(chunks, dim=0)
        if t.shape != original_shape:
            t = t.reshape(original_shape)
        return t

    def __repr__(self) -> str:
        bw_str = {f"{s}->{d}": f"{bw:.1f}GB/s" for (s, d), bw in self._ema_bw.items()}
        return (
            f"HeterogeneousP2PManager("
            f"target_latency={self.target_latency_ms}ms, "
            f"links={bw_str})"
        )


# Standard DES-LOC 5-stage layout constants.
# Ranks 0 and 4 are H100 (fast); ranks 1,2,3 are A6000 (slow).
PP5_DESLOC_FAST_RANKS: Set[int] = {0, 4}
PP5_DESLOC_SLOW_RANKS: Set[int] = {1, 2, 3}


class HeterogeneousBubbleFiller:
    """Opt-in bubble filler for DES-LOC H100+A6000 heterogeneous pipelines.

    Fast ranks (H100) schedule extra forward microbatches during pipeline
    bubbles to increase their utilization from ~40% toward ~70-80%.

    Conceptual overview (PP=5: H100 at ranks 0,4 + A6000 at ranks 1,2,3)
    ----------------------------------------------------------------------
    Standard 1F1B for fast rank 0 (PP=5, M microbatches):

        Warmup:  F0  F1  F2  F3        <- PP-rank-1 = 4 forward passes
        Steady:  [F4,B0] [F5,B1] ...
        Cooldown: B(M-4) B(M-3) B(M-2) B(M-1)

    Fast rank 0 finishes its forward in ~60 ms while A6000 ranks take ~150 ms.
    During the gaps, rank 0 would be idle waiting for the pipeline clock.
    This class detects those gaps and fills them with speculative forward
    microbatches from a shared prefetch queue.

    Attach to config::

        config.desloc = SimpleNamespace(
            bubble_filler=HeterogeneousBubbleFiller(
                fast_ranks={0, 4}, a6000_ranks={1, 2, 3}
            )
        )

    Memory safety
    -------------
    ``activation_memory_budget_mb`` (default 8 GiB) controls how many
    speculative activations may be held simultaneously.  The filler tracks
    an approximate byte count and stops prefetching when the budget is hit.

    Args:
        fast_ranks:                  Set of PP ranks on H100 (fast) GPUs.
        a6000_ranks:                 Set of PP ranks on A6000 (slow) GPUs.
        extra_microbatches:          Max extra microbatches to pre-compute per
                                     bubble (default 2).
        activation_memory_budget_mb: Soft cap on speculative activation memory
                                     in megabytes (default 8192 = 8 GiB).
        alpha:                       EMA smoothing for StageClock (default 0.25).
        initial_fast_ms:             Initial EMA seed for fast ranks (ms).
        initial_slow_ms:             Initial EMA seed for slow ranks (ms).
    """

    def __init__(
        self,
        fast_ranks: Set[int],
        a6000_ranks: Optional[Set[int]] = None,
        extra_microbatches: int = 2,
        activation_memory_budget_mb: int = 8192,
        alpha: float = 0.25,
        initial_fast_ms: float = 60.0,
        initial_slow_ms: float = 150.0,
    ):
        self.fast_ranks: Set[int] = set(fast_ranks)
        self.a6000_ranks: Set[int] = set(a6000_ranks) if a6000_ranks else set()
        self.extra_microbatches = max(1, extra_microbatches)
        self.activation_memory_budget_bytes = activation_memory_budget_mb * 1024 * 1024

        # Per-rank StageClock instances
        all_ranks = self.fast_ranks | self.a6000_ranks
        self.stage_clocks: Dict[int, StageClock] = {}
        for r in all_ranks:
            init_ms = initial_fast_ms if r in self.fast_ranks else initial_slow_ms
            self.stage_clocks[r] = StageClock(alpha=alpha, initial_ms=init_ms)

        # Asymmetric clock scheduler
        num_stages = len(all_ranks) if all_ranks else 5
        self.clock_scheduler = AsymmetricClockScheduler(
            num_stages=num_stages,
            stage_clocks=self.stage_clocks,
            max_outstanding_activations=num_stages - 1,
            fast_rank_set=self.fast_ranks,
        )

        # PCIe-aware P2P manager for this cluster layout.
        # Use is_cross_numa_transfer() to determine whether adjacent ranks
        # are on different NUMA nodes; cross-NUMA links get the conservative
        # PCIe bandwidth estimate (16 GB/s), intra-NUMA links use NVLink
        # bandwidth (400 GB/s for fast↔fast, 32 GB/s for fast↔slow intra-NUMA).
        if all_ranks and len(all_ranks) > 1:
            sorted_ranks = sorted(all_ranks)
            pipeline_pairs_bw: Dict[Tuple[int, int], float] = {}
            for i in range(len(sorted_ranks) - 1):
                s, d = sorted_ranks[i], sorted_ranks[i + 1]
                s_fast = s in self.fast_ranks
                d_fast = d in self.fast_ranks
                # Try to read actual NUMA topology; fall back to tier-based estimate.
                try:
                    cross_numa = is_cross_numa_transfer(s, d)
                except Exception:
                    cross_numa = not (s_fast and d_fast)  # conservative: assume cross-NUMA for mixed
                if cross_numa:
                    bw = 16.0   # PCIe x16 gen4 ~16 GB/s cross-NUMA
                elif s_fast and d_fast:
                    bw = 400.0  # H100↔H100 NVLink C2C
                else:
                    bw = 32.0   # A6000↔A6000 PCIe intra-NUMA (~2× cross-NUMA due to shared root complex)
                pipeline_pairs_bw[(s, d)] = bw
                pipeline_pairs_bw[(d, s)] = bw
            self.p2p_manager = HeterogeneousP2PManager(
                link_bandwidths_gbps=pipeline_pairs_bw,
                target_latency_ms=20.0,
            )
        else:
            self.p2p_manager = HeterogeneousP2PManager()

        # Runtime state
        self._pending_fwd_data: List = []
        self._speculative_activations: List = []   # (input_t, output_t) pairs
        self._approx_bytes_in_flight: int = 0
        self._num_speculative: int = 0

    # ------------------------------------------------------------------
    # Activation memory accounting helpers
    # ------------------------------------------------------------------

    def _tensor_bytes(self, t) -> int:
        """Approximate byte footprint of a tensor or nested structure."""
        if t is None:
            return 0
        if isinstance(t, torch.Tensor):
            return t.numel() * t.element_size()
        if isinstance(t, (list, tuple)):
            return sum(self._tensor_bytes(x) for x in t)
        if isinstance(t, dict):
            return sum(self._tensor_bytes(v) for v in t.values())
        return 0

    def _budget_available(self, candidate_bytes: int = 0) -> bool:
        """Return True if adding candidate_bytes stays within the memory budget."""
        return (
            self._approx_bytes_in_flight + candidate_bytes
        ) < self.activation_memory_budget_bytes

    # ------------------------------------------------------------------
    # Stage clock update API (called by the schedule loop)
    # ------------------------------------------------------------------

    def record_compute_start(self, pp_rank: int) -> None:
        """Mark the start of a compute step on pp_rank."""
        if pp_rank in self.stage_clocks:
            self.stage_clocks[pp_rank].start()

    def record_compute_stop(self, pp_rank: int) -> float:
        """Mark the end of a compute step on pp_rank; return elapsed ms."""
        if pp_rank in self.stage_clocks:
            return self.stage_clocks[pp_rank].stop()
        return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_fast_rank(self, pp_rank: int) -> bool:
        return pp_rank in self.fast_ranks

    def maybe_fill_bubble(
        self,
        pp_rank: int,
        forward_data_store: list,
        config,
        forward_step_func=None,
        data_iterator=None,
        model=None,
        num_microbatches: int = 0,
        speculative_mb_start: int = 0,
    ) -> int:
        """Attempt to fill the pipeline bubble with extra forward passes.

        Called at the two bubble points in
        ``forward_backward_pipelining_without_interleaving``:
          1. After the warmup loop finishes (largest bubble).
          2. Before the cooldown backward passes begin.

        The filler pre-computes up to ``extra_microbatches`` extra forward
        passes from the data iterator on fast ranks (H100), keeping the GPU
        busy while slow ranks (A6000) catch up.  All speculative activations
        are stored in ``_speculative_activations`` and can be accessed later
        via ``pop_speculative_activation()`` to run their backward passes.

        Args:
            pp_rank:              Current PP rank.
            forward_data_store:   List to append loss data to.
            config:               ModelParallelConfig (or None).
            forward_step_func:    User's forward step function (or None to skip).
            data_iterator:        Data iterator for speculative microbatches.
            model:                Model for this stage (or None to skip).
            num_microbatches:     Total microbatches in the global batch.
            speculative_mb_start: Index of the first speculative microbatch.

        Returns:
            Number of speculative microbatches actually computed (0 if fast
            rank conditions are not met or budget is exhausted).
        """
        if not self.is_fast_rank(pp_rank):
            return 0
        if forward_step_func is None or model is None or data_iterator is None:
            return 0

        # Check if slowdown ratio justifies bubble filling
        bottleneck = self.clock_scheduler._find_bottleneck()
        if bottleneck is None or bottleneck == pp_rank:
            return 0
        ratio = self.clock_scheduler.slowdown_ratio(pp_rank, bottleneck)
        if ratio <= 1.15:
            # Stages are close in speed — not worth the overhead
            return 0

        remaining = num_microbatches - speculative_mb_start
        max_extra = min(self.extra_microbatches, remaining)
        if max_extra <= 0:
            return 0

        # Lazy import to avoid circular dependency at module load time
        try:
            from deepspeed.core.pipeline_parallel.schedules import (
                forward_step as _forward_step,
            )
        except ImportError:
            return 0

        hidden_size = getattr(config, 'hidden_size', 4096) if config is not None else 4096
        mbs = getattr(config, 'micro_batch_size', 1) if config is not None else 1
        # Rough size estimate for budget check: seq×batch×hidden×bytes_per_elem
        est_bytes_per_mb = hidden_size * mbs * 4  # float32

        num_computed = 0
        for mb_idx in range(speculative_mb_start, speculative_mb_start + max_extra):
            if not self._budget_available(est_bytes_per_mb):
                break  # Memory budget exceeded — stop prefetching

            self.record_compute_start(pp_rank)
            try:
                output_tensor, num_tokens = _forward_step(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=num_microbatches,
                    input_tensor=None,   # first PP stage: no upstream tensor
                    forward_data_store=forward_data_store,
                    config=config,
                    collect_non_loss_data=False,
                    is_first_microbatch=False,
                    current_microbatch=mb_idx,
                )
            except StopIteration:
                # Data exhausted — no more speculative microbatches available
                break
            except Exception:
                # Don't let speculative compute crash the main training loop
                break
            finally:
                self.record_compute_stop(pp_rank)

            actual_bytes = self._tensor_bytes(output_tensor)
            self._approx_bytes_in_flight += actual_bytes
            self._speculative_activations.append((None, output_tensor))
            self._num_speculative += 1
            num_computed += 1

        return num_computed

    def drain(self, forward_data_store: list, config) -> None:
        """Flush any pending speculative forward outputs into the main store."""
        if self._pending_fwd_data:
            forward_data_store.extend(self._pending_fwd_data)
            self._pending_fwd_data.clear()

    def pop_speculative_activation(self):
        """Pop the oldest (input_tensor, output_tensor) speculative pair.

        Returns None if no speculative activations are available.
        """
        if not self._speculative_activations:
            return None
        pair = self._speculative_activations.pop(0)
        freed = self._tensor_bytes(pair[1])
        self._approx_bytes_in_flight = max(0, self._approx_bytes_in_flight - freed)
        self._num_speculative -= 1
        return pair

    def reset(self) -> None:
        """Clear all speculative state (call at the start of each global step)."""
        self._pending_fwd_data.clear()
        self._speculative_activations.clear()
        self._approx_bytes_in_flight = 0
        self._num_speculative = 0

    def warmup_count_for_rank(self, pp_rank: int, base_warmup: int) -> int:
        """Return (possibly extended) warmup microbatch count for pp_rank.

        Fast ranks may receive extra warmup microbatches to pre-fill the
        pipeline; slow ranks always get the standard base_warmup value.
        """
        return self.clock_scheduler.compute_warmup_override(pp_rank, base_warmup)

    def __repr__(self) -> str:
        clock_summary = {r: f"{c.estimate():.1f}ms" for r, c in self.stage_clocks.items()}
        return (
            f"HeterogeneousBubbleFiller("
            f"fast={self.fast_ranks}, "
            f"slow={self.a6000_ranks}, "
            f"extra_mb={self.extra_microbatches}, "
            f"budget_mb={self.activation_memory_budget_bytes // 1024 // 1024}, "
            f"in_flight={self._num_speculative}, "
            f"clocks={clock_summary})"
        )


# ===========================================================================
# PP=1: No pipeline parallelism
# ===========================================================================

def forward_backward_no_pipelining(
    *, forward_step_func,
    data_iterator, model, num_microbatches,
    seq_length, micro_batch_size,
    decoder_seq_length=None, forward_only=False,
    collect_non_loss_data=False, first_val_step=None,
    adjust_tensor_shapes_fn=None, p2p_communicator=None,
    pg_collection=None, force_all_reduce=False,
):
    """Run forward and backward passes with PP=1.

    Commits: M2280/M2459 cudagraph, M2812 CP loss, M3018 FGAO, M3030/M3213 MoE overlap,
    M3047 hybrid CP, M3544 relaxed assertions, M3734 AG reset, M3977 simplified cudagraph,
    M3981 tp_dp_cp, M4012 paged stash, M4063 early tensor delete.
    """
    if pg_collection is None:
        tp_group = _ps.get_tensor_model_parallel_group()
        cp_group = _ps.get_context_parallel_group()
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = _ps.get_embedding_group(check_initialized=False)
        pg_collection.pos_embd = _ps.get_position_embedding_group(check_initialized=False)
        pg_collection.pp = _ps.get_pipeline_model_parallel_group()
        pg_collection.dp_cp = _ps.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=False)
        pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
    else:
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"

    if isinstance(model, list):
        assert len(model) == 1
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1
        data_iterator = data_iterator[0]
    assert adjust_tensor_shapes_fn is None

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    no_sync_func = config.no_sync_func or contextlib.nullcontext
    model_type = get_model_type(model)
    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if getattr(config, 'overlap_moe_expert_parallel_comm', False) and not forward_only:
        forward_data_store, total_num_tokens = combined_1f1b_schedule_for_no_pipelining(
            forward_step_func, data_iterator, model, num_microbatches,
            input_tensor, output_tensor_grad, forward_data_store, config,
            collect_non_loss_data, first_val_step, forward_only, no_sync_func,
            total_num_tokens, partial(check_first_val_step, first_val_step, forward_only),
        )
    elif getattr(config, 'hybrid_context_parallel', False):
        forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
            forward_step_func, data_iterator, model, num_microbatches,
            input_tensor, output_tensor_grad, forward_data_store, config,
            collect_non_loss_data, first_val_step, forward_only, no_sync_func,
            total_num_tokens, check_first_val_step, model_type,
        )
    else:
        with no_sync_func():
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step(
                    forward_step_func, data_iterator, model, num_microbatches,
                    input_tensor, forward_data_store, config, pg_collection.cp.size(),
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                )
                total_num_tokens += num_tokens
                if not forward_only:
                    backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                    del output_tensor  # M4063: release before next forward to avoid stream warning

        output_tensor, num_tokens = forward_step(
            forward_step_func, data_iterator, model, num_microbatches,
            input_tensor, forward_data_store, config, pg_collection.cp.size(),
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, num_microbatches == 1),
            current_microbatch=num_microbatches - 1,
        )
        total_num_tokens += num_tokens
        if not forward_only:
            backward_step(input_tensor, output_tensor, output_tensor_grad, config)
            del output_tensor  # M4063

    if config.finalize_model_grads_func is not None and not forward_only:
        total_num_tokens = torch.clamp(total_num_tokens, min=1)  # From Megatron M3531: guard all-padding batches
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection, force_all_reduce=force_all_reduce,
        )

    if _HAS_FGAO and getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    # M3734: Reset AG pipeline bucket before next validation iteration
    if forward_only:
        for mc in [model]:
            if (hasattr(mc, 'ddp_config') and mc.ddp_config.use_megatron_fsdp
                    and mc.ddp_config.overlap_param_gather):
                mc.synchronize_param_gather()

    if config.timers is not None:
        config.timers('forward-backward').stop()
    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()
    return forward_data_store


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    config = get_model_config(model[0])
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=_ps.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = _ps.get_tensor_model_parallel_group()
        cp_group = _ps.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = _ps.get_embedding_group(check_initialized=False)
        pp_group = _ps.get_pipeline_model_parallel_group()
        pos_emb_group = _ps.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = _ps.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for forward_step_helper"""
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # pylint: disable=E0606
        if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        return input_tensor, output_tensor, output_tensor_grad

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        if config.overlap_moe_expert_parallel_comm and not forward_only:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)

        if config.overlap_p2p_comm_warmup_flush:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
            # isend() copies asynchronously; wait until the copy is done before
            # freeing the source buffer, otherwise the next PP stage gets corrupted data.
            if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
                send_next_wait_handle.wait()
                send_next_wait_handle = None

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        if config.overlap_p2p_comm:

            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # isend() copies asynchronously; wait until the copy is done before
                # freeing the source buffer, otherwise the next PP stage gets corrupted data.
                if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
                    send_next_wait_handle.wait()
                    send_next_wait_handle = None
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communicator.recv_backward(
                    tensor_shape,
                    is_last_stage=(
                        _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                    ),
                )
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not is_pp_last_stage(p2p_communicator.pp_group):
                    _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communicator.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).

        total_num_tokens = torch.clamp(total_num_tokens, min=1)  # From Megatron M3531: guard all-padding batches
        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if _HAS_FGAO and getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()
    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    return forward_data_store

def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    stage: Optional[int] = None,
):
    """Determine tensor shapes for pipeline communication.

    Returns [()] for variable_seq_lengths mode (shapes exchanged dynamically),
    or computed shapes for fixed sequence length mode.

    DES-LOC heterogeneous PP=5 extension:
        When ``config.hetero_micro_batch_sizes`` is set and ``stage`` is
        provided, the micro-batch size for *that* stage is used instead of
        the global ``micro_batch_size``.  This allows fast stages (H100) to
        process larger micro-batches while slow stages (A6000) use smaller
        ones, reducing pipeline bubbles imposed by the slowest stage.

        When stages use different micro-batch sizes, ``config.variable_seq_lengths``
        is automatically treated as True for P2P shape negotiation so that the
        receiver allocates a buffer matching the sender's actual tensor size.
    """
    tensor_shapes = []

    # Heterogeneous micro-batch size: if per-stage sizes are configured and
    # differ from the global default, force variable-shape communication so
    # adjacent stages can negotiate the correct tensor dimensions at runtime.
    hetero_mbs = getattr(config, 'hetero_micro_batch_sizes', None)
    if hetero_mbs is not None and stage is not None:
        effective_mbs = config.get_stage_micro_batch_size(stage, micro_batch_size)
        # If any stage differs in micro-batch size, shapes must be negotiated
        # dynamically — fall through to variable_seq_lengths path.
        if effective_mbs != micro_batch_size or len(set(hetero_mbs)) > 1:
            tensor_shapes.append(())
            return tensor_shapes
        micro_batch_size = effective_mbs

    if config.variable_seq_lengths:
        # Shapes exchanged dynamically during P2P communication
        tensor_shapes.append(())
        return tensor_shapes

    # Fixed sequence lengths - compute shape
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    if cp_group is not None:
        effective_seq_length = effective_seq_length // cp_group.size()

    if config.sequence_parallel and tp_group is not None:
        effective_seq_length = effective_seq_length // tp_group.size()

    tensor_shapes.append((effective_seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes

def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    # Determine if this is a multi-module pipeline
    # (used for validation and backward function selection)
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    )

    if p2p_communicator is None and pg_collection is None:
        # Default: single-module with parallel_state groups
        p2p_communicator = P2PCommunicator(
            pp_group=_ps.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = _ps.get_tensor_model_parallel_group()
        cp_group = _ps.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = _ps.get_embedding_group(check_initialized=False)
        pos_emb_group = _ps.get_position_embedding_group(check_initialized=False)
        pp_group = _ps.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = _ps.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"

        if is_multimodule:
            # Multi-module: use language model's CP size for loss scaling
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            if pg_collection.has_language_model():
                cp_size = pg_collection.get_language_model_cp_size()
            else:
                # Encoder-only ranks should not use CP loss scaling.
                cp_size = None

        elif isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()

        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

    # DES-LOC: let the bubble filler extend the warmup count for fast ranks (H100).
    # Fast ranks pre-compute extra microbatches during the warmup phase so that
    # more activations are ready in SRAM when the slow stages catch up, reducing
    # the effective steady-state bubble.  The extension is capped at
    # min(num_microbatches, total_stages - 1) so we never exceed the activation
    # memory budget.
    _desloc = getattr(config, 'desloc', None)
    _bubble_filler: Optional["HeterogeneousBubbleFiller"] = getattr(_desloc, 'bubble_filler', None)
    _current_pp_rank = p2p_communicator.current_stage
    if _bubble_filler is not None:
        _extended_warmup = _bubble_filler.warmup_count_for_rank(
            _current_pp_rank, num_warmup_microbatches
        )
        # Cap at num_microbatches so we never run more warmups than microbatches
        num_warmup_microbatches = min(_extended_warmup, num_microbatches)

    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Select backward function based on whether multi-module or single-module
    if is_multimodule:
        backward_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_func = backward_step

    # DES-LOC: current PP stage index for heterogeneous micro-batch sizing
    _current_pp_stage = p2p_communicator.current_stage

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        stage=_current_pp_stage,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        stage=_current_pp_stage,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

        # DES-LOC: time each warmup forward pass to calibrate the stage clock
        if _bubble_filler is not None:
            _bubble_filler.record_compute_start(_current_pp_rank)

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )

        if _bubble_filler is not None:
            _bubble_filler.record_compute_stop(_current_pp_rank)

        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # DES-LOC: fill the post-warmup bubble on fast ranks (H100).
    # This is the largest bubble in the 1F1B schedule — (PP-1) * slow_stage_time.
    # Fast ranks can pre-compute extra microbatches here while slow ranks are
    # still finishing their warmup passes.
    if _bubble_filler is not None and not forward_only:
        _bubble_filler.maybe_fill_bubble(
            pp_rank=_current_pp_rank,
            forward_data_store=forward_data_store,
            config=config,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            speculative_mb_start=num_warmup_microbatches,
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        # DES-LOC: time each steady-state forward pass to keep stage-clock EMA current
        if _bubble_filler is not None:
            _bubble_filler.record_compute_start(_current_pp_rank)

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )

        if _bubble_filler is not None:
            _bubble_filler.record_compute_stop(_current_pp_rank)
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        total_num_tokens = torch.clamp(total_num_tokens, min=1)  # From Megatron M3531: guard all-padding batches
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if _HAS_FGAO and getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()

    # DES-LOC: flush any pending speculative forward data from the bubble filler
    if _bubble_filler is not None:
        _bubble_filler.drain(forward_data_store, config=None)

    return forward_data_store

def set_pipeline_layer_split(split: List[int]) -> None:
    """Register per-stage layer counts for heterogeneous (DES-LOC) pipelines."""
    global _PIPELINE_LAYER_SPLIT
    if not split or any(c <= 0 for c in split):
        raise ValueError(f"pipeline_layer_split must be positive ints, got {split}")
    _PIPELINE_LAYER_SPLIT = list(split)


def get_pipeline_model_parallel_rank_for_layer(layer_number: int) -> int:
    """Return which PP rank owns a given 0-based global layer index.
    
    DES-LOC example (5-stage split [4,8,8,4,8] = 32 layers total):
        Layers  0-3  -> rank 0
        Layers 4-11  -> rank 1
        Layers 12-19 -> rank 2
        Layers 20-23 -> rank 3
        Layers 24-31 -> rank 4
    """
    split = None
    if _ps is not None:
        split = getattr(_ps, "_PIPELINE_LAYER_SPLIT", None)
    if split is None:
        split = _PIPELINE_LAYER_SPLIT
    if split is None:
        if _ps is None or not torch.distributed.is_initialized():
            raise RuntimeError("pipeline_layer_split not configured and distributed not initialised")
        split = [1] * _ps.get_pipeline_model_parallel_world_size()
    cumulative = 0
    for rank, count in enumerate(split):
        cumulative += count
        if layer_number < cumulative:
            return rank
    raise ValueError(
        f"layer_number {layer_number} out of range for split {split} (total {cumulative})"
    )


def get_num_microbatches() -> int:
    """Return current global microbatch count from parallel_state."""
    if _ps is None:
        raise RuntimeError("parallel_state is not available")
    fn = getattr(_ps, "get_num_microbatches", None)
    if fn is None:
        raise AttributeError("parallel_state does not expose get_num_microbatches")
    return fn()


# ===========================================================================
# DES-LOC: PP=5 heterogeneous 1F1B wrapper
# ===========================================================================

def forward_backward_pipelining_without_interleaving_pp5_heterogeneous(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional["P2PCommunicator"] = None,
    pg_collection=None,
    force_all_reduce: Optional[bool] = False,
    bubble_filler: Optional["HeterogeneousBubbleFiller"] = None,
):
    """1F1B schedule for PP=5 heterogeneous (H100+A6000) pipelines.

    This is a thin wrapper around ``forward_backward_pipelining_without_interleaving``
    that plugs in the ``HeterogeneousBubbleFiller`` at the two natural bubble
    points (post-warmup and pre-cooldown) so that fast stages (H100) compute
    extra microbatches while slow stages (A6000) catch up.

    PP=5 bubble analysis
    --------------------
    For PP=5 with M microbatches:
      - Standard warmup: ranks [0,1,2,3,4] perform [4,3,2,1,0] forward passes.
      - Bubble fraction: (PP-1)/M = 4/M.
      - With HeterogeneousBubbleFiller (extra_mb=2): fast ranks do 2 additional
        forward passes during the warmup gap, improving utilization from ~40%
        to ~60% on H100 ranks.

    Asymmetric speed example
    ------------------------
    Rank 0 (H100, 60ms/mb) waits for rank 1 (A6000, 150ms/mb):
        Bubble = (PP-1) * slow_ms = 4 * 150 = 600ms per microbatch cycle
        Fillable by fast rank = 2 extra * 60ms = 120ms, ~20% utilization gain

    Args:
        bubble_filler: A ``HeterogeneousBubbleFiller`` instance.  If None,
                       falls back to the standard 1F1B schedule identically.
        (other args):  Same as ``forward_backward_pipelining_without_interleaving``.

    Returns:
        forward_data_store (list of loss data from last stage).
    """
    # Determine current PP rank for bubble-filler decisions
    current_pp_rank = 0
    if p2p_communicator is not None:
        current_pp_rank = p2p_communicator.current_stage
    elif _ps is not None:
        try:
            if torch.distributed.is_initialized():
                current_pp_rank = _ps.get_pipeline_model_parallel_rank()
        except Exception:
            pass

    # Reset bubble filler state for this global step
    if bubble_filler is not None:
        bubble_filler.reset()

    # Time the overall step on fast ranks so the clock learns stage speed
    if bubble_filler is not None and bubble_filler.is_fast_rank(current_pp_rank):
        bubble_filler.record_compute_start(current_pp_rank)

    # --- Delegate to the standard non-interleaved 1F1B schedule ---
    result = forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        forward_only=forward_only,
        collect_non_loss_data=collect_non_loss_data,
        first_val_step=first_val_step,
        adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        p2p_communicator=p2p_communicator,
        pg_collection=pg_collection,
        force_all_reduce=force_all_reduce,
    )

    if bubble_filler is not None and bubble_filler.is_fast_rank(current_pp_rank):
        # Stop the overall timing clock and update the EMA
        bubble_filler.record_compute_stop(current_pp_rank)

        if not forward_only:
            # Attempt to fill the residual post-step bubble.
            # speculative_mb_start=num_microbatches means "beyond the batch" —
            # the filler will detect num_microbatches_remaining==0 and return 0,
            # which is the correct guard for the first step before clocks warm up.
            # After a few steps the EMA ratios will reflect actual GPU speeds
            # and the filler will engage when ratio > 1.15.
            cfg = None
            if p2p_communicator is not None:
                cfg = getattr(p2p_communicator, 'config', None)
            bubble_filler.maybe_fill_bubble(
                pp_rank=current_pp_rank,
                forward_data_store=result,
                config=cfg,
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                speculative_mb_start=num_microbatches,  # guard: no extra data
            )

        bubble_filler.drain(result, config=None)

    return result


# ===========================================================================
# DES-LOC: heterogeneous 1F1B schedule (native, non-wrapping)
# ===========================================================================

def forward_backward_hetero_1f1b(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional["P2PCommunicator"] = None,
    pg_collection=None,
    force_all_reduce: Optional[bool] = False,
    bubble_filler: Optional["HeterogeneousBubbleFiller"] = None,
):
    """Native heterogeneous 1F1B schedule for DES-LOC H100+A6000 pipelines.

    Unlike ``forward_backward_pipelining_without_interleaving_pp5_heterogeneous``
    (which wraps the standard schedule and fills bubbles only post-hoc), this
    function integrates heterogeneous-aware behaviour directly into the warmup,
    steady-state, and cooldown phases:

    Phase 0 — Asymmetric warmup
    ----------------------------
    Each rank executes a *rank-local* number of warmup microbatches.  For fast
    ranks (H100, ranks 0 and 4 in the standard DES-LOC PP=5 layout), the
    ``HeterogeneousBubbleFiller`` can extend the warmup count beyond the
    standard ``PP - rank - 1`` value so that more activations are already
    resident in SRAM when the slow stages (A6000) catch up.  Slow ranks always
    use the standard warmup depth.

    Phase 1 — Steady-state 1F1B with inline bubble filling
    --------------------------------------------------------
    After the warmup, each rank enters the interleaved forward-backward (1F1B)
    steady state.  At the transition point (the "post-warmup bubble") the
    ``HeterogeneousBubbleFiller.maybe_fill_bubble`` is called on fast ranks to
    schedule extra speculative forward microbatches.  Per-microbatch timing is
    recorded so the stage-clock EMA stays current throughout.

    Phase 2 — Cooldown with speculative drain
    ------------------------------------------
    Cooldown backward passes consume any speculative activations queued by the
    bubble filler (via ``pop_speculative_activation``) before falling back to
    the standard ``input_tensors`` / ``output_tensors`` stacks.  This ensures
    no speculative activation is orphaned and all computed gradients are
    properly synchronised before the function returns.

    DES-LOC bubble analysis (PP=5, M microbatches)
    ------------------------------------------------
    Standard 1F1B bubble fraction: (PP-1)/M = 4/M.
    With extended warmup (extra_mb=2 on H100 ranks): effective bubble ≈ 2/M.
    Additional inline fill during steady-state can further cut idle time on
    fast ranks by ~15-20% depending on slowdown_ratio.

    Args:
        forward_step_func:    User's per-microbatch forward step.
        data_iterator:        Data iterator (or list for interleaved).
        model:                Model module (or list for interleaved).
        num_microbatches:     Number of microbatches in the global batch.
        seq_length:           Sequence length for the current global batch.
        micro_batch_size:     Number of sequences per microbatch.
        decoder_seq_length:   Decoder sequence length (dual-stack only).
        forward_only:         Skip all backward passes.
        collect_non_loss_data: Passed through to forward_step.
        first_val_step:       First validation step flag (TE FP8 update).
        adjust_tensor_shapes_fn: Optional recv/send shape adjuster.
        p2p_communicator:     P2P communicator; if None, built from parallel_state.
        pg_collection:        Process-group collection; if None, built from parallel_state.
        force_all_reduce:     Force all-reduce instead of reduce-scatter.
        bubble_filler:        ``HeterogeneousBubbleFiller`` instance.  If None
                              the schedule degrades gracefully to standard 1F1B.

    Returns:
        forward_data_store: List of (loss, loss_dict) pairs from the last PP stage.
    """
    # ------------------------------------------------------------------
    # Normalise model / data_iterator
    # ------------------------------------------------------------------
    if isinstance(model, list):
        assert len(model) == 1, (
            "hetero_1f1b does not support model chunking (VPP); "
            "use forward_backward_pipelining_with_interleaving instead"
        )
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, (
            "hetero_1f1b does not support multiple data iterators"
        )
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "hetero_1f1b does not support overlap_p2p_comm; "
            "disable config.overlap_p2p_comm or use the standard schedule"
        )

    # ------------------------------------------------------------------
    # Process-group / P2P setup (mirrors standard schedule)
    # ------------------------------------------------------------------
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    )
    tp_group, cp_group, cp_size = None, None, None

    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=_ps.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = _ps.get_tensor_model_parallel_group()
        cp_group = _ps.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = _ps.get_embedding_group(check_initialized=False)
        pos_emb_group = _ps.get_position_embedding_group(check_initialized=False)
        pp_group = _ps.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = _ps.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )
    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        if is_multimodule:
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            cp_size = pg_collection.get_language_model_cp_size() if pg_collection.has_language_model() else None
        elif isinstance(pg_collection, ProcessGroupCollection):
            assert hasattr(pg_collection, 'tp') and hasattr(pg_collection, 'cp')
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()
        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # ------------------------------------------------------------------
    # Embedding / grad-sync setup
    # ------------------------------------------------------------------
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # ------------------------------------------------------------------
    # Hetero-aware warmup count
    # ------------------------------------------------------------------
    _current_pp_rank = p2p_communicator.current_stage

    # Standard 1F1B warmup: (total_stages - current_stage - 1) forward passes.
    base_warmup = p2p_communicator.total_stages - _current_pp_rank - 1
    base_warmup = min(base_warmup, num_microbatches)

    # Reset bubble filler state for this global step.
    if bubble_filler is not None:
        bubble_filler.reset()

    # Fast ranks may get an extended warmup to pre-fill the pipeline while
    # slow ranks are still computing their first warmup microbatches.
    if bubble_filler is not None:
        extended_warmup = bubble_filler.warmup_count_for_rank(_current_pp_rank, base_warmup)
        num_warmup_microbatches = min(extended_warmup, num_microbatches)
    else:
        num_warmup_microbatches = base_warmup

    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Activation-checkpoint accounting (mirrors standard schedule).
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Backward function selection
    if is_multimodule:
        backward_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_func = backward_step

    # Tensor shape negotiation
    _current_pp_stage = p2p_communicator.current_stage
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        stage=_current_pp_stage,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        stage=_current_pp_stage,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    input_tensors: Optional[List] = None
    output_tensors: Optional[List] = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store: List = []

    # ------------------------------------------------------------------
    # Phase 0: Asymmetric warmup
    # ------------------------------------------------------------------
    # Start the overall compute clock on fast ranks so the EMA has a baseline
    # for the first bubble-fill decision.
    if bubble_filler is not None and bubble_filler.is_fast_rank(_current_pp_rank):
        bubble_filler.record_compute_start(_current_pp_rank)

    for i in range(num_warmup_microbatches):
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

        # Per-microbatch timing for stage-clock EMA calibration.
        if bubble_filler is not None:
            bubble_filler.record_compute_start(_current_pp_rank)

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )

        if bubble_filler is not None:
            bubble_filler.record_compute_stop(_current_pp_rank)

        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # ------------------------------------------------------------------
    # Post-warmup bubble fill (largest bubble in the pipeline)
    # ------------------------------------------------------------------
    # Receive the first steady-state input tensor *before* filling the bubble
    # so the pipeline stays unblocked: the recv is non-blocking on fast ranks.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    if bubble_filler is not None and not forward_only:
        # On fast ranks the warmup has just completed and slow stages are
        # still in their warmup — this is the (PP-1)*slow_ms bubble.
        # Fill it with speculative forward microbatches.
        bubble_filler.maybe_fill_bubble(
            pp_rank=_current_pp_rank,
            forward_data_store=forward_data_store,
            config=config,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            speculative_mb_start=num_warmup_microbatches,
        )

    # ------------------------------------------------------------------
    # Phase 1: Steady-state 1F1B
    # ------------------------------------------------------------------
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        if bubble_filler is not None:
            bubble_filler.record_compute_start(_current_pp_rank)

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )

        if bubble_filler is not None:
            bubble_filler.record_compute_stop(_current_pp_rank)
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # ------------------------------------------------------------------
    # Phase 2: Cooldown — backward passes + speculative activation drain
    # ------------------------------------------------------------------
    if not forward_only:
        # Total backward passes = warmup count + any speculative activations
        # queued by the bubble filler.  We drain speculative activations first
        # (oldest → newest) so their gradients are computed in the same order
        # as the forward passes, preserving numerical equivalence with the
        # standard schedule.
        speculative_pairs: List = []
        if bubble_filler is not None:
            while True:
                pair = bubble_filler.pop_speculative_activation()
                if pair is None:
                    break
                speculative_pairs.append(pair)

        # Number of standard cooldown backward passes (warmup activations).
        n_standard_cooldown = num_warmup_microbatches
        # Number of speculative backward passes.
        n_speculative_cooldown = len(speculative_pairs)
        total_cooldown = n_standard_cooldown + n_speculative_cooldown

        for i in range(total_cooldown):
            is_last_cooldown = i == total_cooldown - 1

            # Enable grad sync on the final backward pass.
            if is_last_cooldown:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            if i < n_standard_cooldown:
                # Standard cooldown: consume activations from the warmup queue.
                in_t = input_tensors.pop(0)
                out_t = output_tensors.pop(0)
            else:
                # Speculative cooldown: consume activations from bubble filler.
                spec_idx = i - n_standard_cooldown
                in_t, out_t = speculative_pairs[spec_idx]

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func(in_t, out_t, output_tensor_grad, config)

            p2p_communicator.send_backward(
                input_tensor_grad, p2p_communicator.is_pp_first_stage
            )

        # Drain any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    # Stop overall timing clock on fast ranks.
    if bubble_filler is not None and bubble_filler.is_fast_rank(_current_pp_rank):
        bubble_filler.record_compute_stop(_current_pp_rank)

    # ------------------------------------------------------------------
    # Gradient finalisation
    # ------------------------------------------------------------------
    if config.finalize_model_grads_func is not None and not forward_only:
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )
        total_num_tokens = torch.clamp(total_num_tokens, min=1)
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if _HAS_FGAO and getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()

    # Final flush of any pending speculative data into the store.
    if bubble_filler is not None:
        bubble_filler.drain(forward_data_store, config=None)

    return forward_data_store


# ===========================================================================
# DES-LOC: PP=5 factory helpers
# ===========================================================================

def make_pp5_bubble_filler(
    extra_microbatches: int = 2,
    activation_memory_budget_mb: int = 8192,
    initial_fast_ms: float = 60.0,
    initial_slow_ms: float = 150.0,
) -> "HeterogeneousBubbleFiller":
    """Factory for the standard DES-LOC PP=5 HeterogeneousBubbleFiller.

    Creates a filler configured for the 2×H100 (ranks 0,4) + 3×A6000
    (ranks 1,2,3) layout used by DES-LOC clusters.

    Args:
        extra_microbatches:          Max extra microbatches per bubble (default 2).
        activation_memory_budget_mb: Activation memory budget in MB (default 8192).
        initial_fast_ms:             Initial compute-time seed for H100 ranks (ms).
        initial_slow_ms:             Initial compute-time seed for A6000 ranks (ms).

    Returns:
        Configured ``HeterogeneousBubbleFiller`` instance.
    """
    return HeterogeneousBubbleFiller(
        fast_ranks=PP5_DESLOC_FAST_RANKS,
        a6000_ranks=PP5_DESLOC_SLOW_RANKS,
        extra_microbatches=extra_microbatches,
        activation_memory_budget_mb=activation_memory_budget_mb,
        initial_fast_ms=initial_fast_ms,
        initial_slow_ms=initial_slow_ms,
    )


def make_pp5_p2p_manager(
    h100_a6000_bw_gbps: float = 16.0,
    h100_h100_bw_gbps: float = 400.0,
    target_latency_ms: float = 20.0,
) -> "HeterogeneousP2PManager":
    """Factory for the standard DES-LOC PP=5 HeterogeneousP2PManager.

    Initialises per-link bandwidth estimates for the standard DES-LOC
    PP=5 ring topology (ranks 0→1→2→3→4):
      - 0↔1 (H100↔A6000): PCIe x16, h100_a6000_bw_gbps
      - 1↔2 (A6000↔A6000): PCIe / limited NVLink, h100_a6000_bw_gbps
      - 2↔3 (A6000↔A6000): same
      - 3↔4 (A6000↔H100): PCIe x16, h100_a6000_bw_gbps

    Args:
        h100_a6000_bw_gbps: Bandwidth for H100<->A6000 links (GB/s).
        h100_h100_bw_gbps:  Bandwidth for H100<->H100 links (GB/s, n/a for PP=5 ring).
        target_latency_ms:  Target one-way transfer latency budget (ms).

    Returns:
        Configured ``HeterogeneousP2PManager`` instance.
    """
    pipeline_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    link_bw: Dict[Tuple[int, int], float] = {}
    for src, dst in pipeline_pairs:
        src_fast = src in PP5_DESLOC_FAST_RANKS
        dst_fast = dst in PP5_DESLOC_FAST_RANKS
        bw = h100_h100_bw_gbps if (src_fast and dst_fast) else h100_a6000_bw_gbps
        link_bw[(src, dst)] = bw
        link_bw[(dst, src)] = bw  # backward pass traverses same physical link
    return HeterogeneousP2PManager(
        link_bandwidths_gbps=link_bw,
        target_latency_ms=target_latency_ms,
        alpha=0.2,
    )


__all__ = [
    # Schedule selector
    "get_forward_backward_func",
    # Step functions
    "forward_step", "forward_step_calc_loss",
    "backward_step", "backward_step_multimodule",
    # Standard schedules
    "forward_backward_no_pipelining",
    "forward_backward_pipelining_without_interleaving",
    "forward_backward_pipelining_with_interleaving",
    # DES-LOC heterogeneous schedules
    "forward_backward_hetero_1f1b",
    "forward_backward_pipelining_without_interleaving_pp5_heterogeneous",
    # DES-LOC bubble filling (full implementation)
    "StageClock",
    "AsymmetricClockScheduler",
    "HeterogeneousBubbleFiller",
    # DES-LOC PCIe-aware P2P manager
    "HeterogeneousP2PManager",
    # DES-LOC PP=5 layout constants
    "PP5_DESLOC_FAST_RANKS",
    "PP5_DESLOC_SLOW_RANKS",
    # DES-LOC factory helpers
    "make_pp5_bubble_filler",
    "make_pp5_p2p_manager",
    # Utilities
    "get_tensor_shapes", "get_pp_rank_microbatches", "get_schedule_table",
    "deallocate_output_tensor", "custom_backward", "get_tensor_device",
    "check_first_val_step", "clear_embedding_activation_buffer",
    "finish_embedding_wgrad_compute", "get_num_microbatches",
    "get_pipeline_model_parallel_rank_for_layer", "set_pipeline_layer_split",
]
