# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Ported from Megatron-LM megatron/core/transformer/multi_token_prediction.py
# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Multi-Token Prediction (MTP) layer and block.

Ported from Megatron-LM ``megatron/core/transformer/multi_token_prediction.py``
(Megatron commits up to M4147+).  All ``megatron.core.*`` imports have been
replaced with their ``deepspeed.core.*`` equivalents.

Key changes vs. upstream
-------------------------
* ``megatron.core``                   → ``deepspeed.core``
* ``megatron.core.InferenceParams``   → ``deepspeed.compile.megatron_forward_step.InferenceParams``
* ``megatron.core.transformer.module.MegatronModule``
                                      → ``deepspeed.core.transformer.module.MegatronModule``
* ``megatron.core.tensor_parallel.*`` → ``deepspeed.core.tensor_parallel.*``
* ``megatron.core.parallel_state``    → ``deepspeed.core.parallel_state``
* ``megatron.core.enums.*``           → ``deepspeed.core.enums.*``
* ``megatron.core.process_groups_config`` → ``deepspeed.core.process_groups_config``
* ``Fp8Recipe``, ``get_fp8_context``, ``HAVE_TE``, TE-specific helpers →
  guarded stubs / removed (FP8 path left as ``nullcontext`` when DS hasn't
  implemented those helpers yet).
* ``InferenceMode.is_active()``       → always False (inference-time gather
  path falls back to training gather).
* ``make_viewless_tensor``            → inline ``_make_viewless_tensor`` helper.
* ``make_tp_sharded_tensor_for_checkpoint``, ``apply_prefix_mapping``,
  ``replace_prefix_for_sharding``     → lightweight stubs that preserve the
  same call-sites so the code runs; replace with real DS dist-ckpt helpers
  when available.
* ``apply_module``                    → identity wrapper (not needed in DS).
* Hybrid/Mamba sub-paths (``HybridStack``, ``TESpecProvider``,
  ``PipelineParallelLayerLayout``, ``BackendSpecProvider``) are kept as
  guarded imports so the GPT/MTP-only path works without those modules.
"""
from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
from torch import Tensor

import deepspeed.core.parallel_state as parallel_state
import deepspeed.core.tensor_parallel as tensor_parallel
from deepspeed.core.enums import AttnMaskType, LayerType
from deepspeed.core.process_groups_config import ProcessGroupCollection
from deepspeed.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Optional / conditional imports
# ---------------------------------------------------------------------------

# InferenceParams lives in deepspeed.compile for now.
try:
    from deepspeed.compile.megatron_forward_step import InferenceParams
except ImportError:  # graceful fallback
    InferenceParams = None  # type: ignore[assignment,misc]

# PackedSeqParams — not yet ported; define a minimal placeholder.
try:
    from deepspeed.core.packed_seq_params import PackedSeqParams
except ImportError:
    @dataclass
    class PackedSeqParams:  # type: ignore[no-redef]
        """Minimal PackedSeqParams placeholder for packed-sequence MTP support."""
        cu_seqlens_q: Optional[Tensor] = None
        cu_seqlens_kv: Optional[Tensor] = None

# ShardedStateDict type alias — plain dict in the DS port.
ShardedStateDict = dict

# dist_checkpointing helpers — lightweight stubs until DS implements them.
def _stub_make_tp_sharded_tensor_for_checkpoint(
    tensor, key, replica_id=None, allow_shape_mismatch=False,
    tp_group=None, dp_cp_group=None
):
    """Stub: return the tensor itself, recording metadata as tensor attributes."""
    tensor._ds_ckpt_key = key
    tensor._ds_ckpt_replica_id = replica_id
    return tensor


def _stub_apply_prefix_mapping(sharded_state_dict: dict, prefix_map: dict) -> None:
    """Stub: rename keys in *sharded_state_dict* according to *prefix_map*."""
    for old_prefix, new_prefix in prefix_map.items():
        keys_to_rename = [k for k in list(sharded_state_dict) if k.startswith(old_prefix)]
        for old_key in keys_to_rename:
            new_key = new_prefix + old_key[len(old_prefix):]
            sharded_state_dict[new_key] = sharded_state_dict.pop(old_key)


def _stub_replace_prefix_for_sharding(
    sharded_state_dict: dict, src_prefix: str, tgt_prefix: str
) -> None:
    """Stub: rename every key that starts with *src_prefix* to use *tgt_prefix*."""
    _stub_apply_prefix_mapping(sharded_state_dict, {src_prefix: tgt_prefix})


make_tp_sharded_tensor_for_checkpoint = _stub_make_tp_sharded_tensor_for_checkpoint
apply_prefix_mapping = _stub_apply_prefix_mapping
replace_prefix_for_sharding = _stub_replace_prefix_for_sharding

# FP8 — not yet fully ported; fall back to nullcontext.
try:
    from deepspeed.core.fp8_utils import get_fp8_context
    HAVE_FP8 = True
except ImportError:
    def get_fp8_context(config, is_init: bool = False):  # type: ignore[misc]
        return nullcontext()
    HAVE_FP8 = False

# fp8_recipe enum — stub if not available.
try:
    from deepspeed.core.enums import Fp8Recipe
except ImportError:
    class Fp8Recipe:  # type: ignore[misc]
        delayed = "delayed"

# Transformer Engine inference gather — not available; always use standard path.
def inference_all_gather_from_tensor_model_parallel_region(hidden, tp_group, config):
    """Stub: redirect to standard TP gather (inference path not yet ported)."""
    return gather_from_tensor_model_parallel_region(hidden, group=tp_group)

# get_pg_rank helper.
try:
    from deepspeed.core.utils import get_pg_rank
except ImportError:
    def get_pg_rank(group) -> int:  # type: ignore[misc]
        return torch.distributed.get_rank(group=group)

# is_torch_min_version helper.
try:
    from deepspeed.core.utils import is_torch_min_version
except ImportError:
    def is_torch_min_version(version: str) -> bool:  # type: ignore[misc]
        from packaging.version import Version
        return Version(torch.__version__) >= Version(version)

# apply_module — in Megatron it handles TE module wrapping; here it's identity.
def apply_module(module):
    """Identity wrapper — TE module delegation not needed in DS port."""
    return module

# Hybrid / Mamba path — optional.
try:
    from deepspeed.core.models.hybrid.hybrid_block import HybridStackSubmodules  # type: ignore
    HAS_HYBRID = True
except ImportError:
    HybridStackSubmodules = None  # type: ignore[assignment,misc]
    HAS_HYBRID = False

# PipelineParallelLayerLayout — optional.
try:
    from deepspeed.core.transformer.pipeline_parallel_layer_layout import (  # type: ignore
        PipelineParallelLayerLayout,
    )
    HAS_PP_LAYOUT = True
except ImportError:
    PipelineParallelLayerLayout = None  # type: ignore[assignment,misc]
    HAS_PP_LAYOUT = False

# is_vp_last_stage helper.
try:
    from deepspeed.core.pipeline_parallel.utils import is_vp_last_stage
except ImportError:
    def is_vp_last_stage(vp_stage, vp_size) -> bool:  # type: ignore[misc]
        if vp_size is None or vp_size <= 1:
            return True
        return vp_stage == (vp_size - 1)

# ModuleSpec / build_module — lightweight shims if DS hasn't ported spec_utils.
try:
    from deepspeed.core.transformer.spec_utils import ModuleSpec, build_module  # type: ignore
except ImportError:
    @dataclass
    class ModuleSpec:  # type: ignore[no-redef]
        module: type
        submodules: object = None
        params: dict = None

        def __post_init__(self):
            if self.params is None:
                object.__setattr__(self, "params", {})

    def build_module(spec, *args, **kwargs):  # type: ignore[misc]
        if isinstance(spec, ModuleSpec):
            return spec.module(*args, **kwargs)
        return spec(*args, **kwargs)

# LayerNormBuilder type alias — the DS TransformerConfig already uses callables
# as norm builders; treat as ``type`` here.
LayerNormBuilder = type

# TransformerBlockSubmodules — for type hints only.
try:
    from deepspeed.core.transformer.transformer_block import TransformerBlockSubmodules
except ImportError:
    TransformerBlockSubmodules = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Inline utilities
# ---------------------------------------------------------------------------

def _make_viewless_tensor(
    inp: Tensor, requires_grad: bool = True, keep_graph: bool = True
) -> Tensor:
    """Return a viewless copy of *inp*.

    Megatron's ``make_viewless_tensor`` prevents schedule.py's
    ``deallocate_output_tensor()`` from tripping on viewed tensors.  In DS we
    replicate that by calling ``.contiguous()`` when the tensor is a view.
    """
    if inp is None:
        return inp
    if inp._base is None:
        # Not a view — ensure requires_grad is as requested and return as-is.
        if requires_grad and not inp.requires_grad:
            inp = inp.requires_grad_(True)
        return inp
    out = inp.contiguous()
    if requires_grad and not out.requires_grad:
        out = out.requires_grad_(True)
    return out


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
]
# Add padding_causal if the DS AttnMaskType enum defines it.
try:
    SUPPORTED_ATTN_MASK.append(AttnMaskType.padding_causal)  # type: ignore[attr-defined]
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Sharded-state-dict helpers
# ---------------------------------------------------------------------------

def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict,
    word_emb_weight: Tensor,
    word_emb_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """Tie the embedding weight of the MTP stage in a sharded state dict."""
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre-processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert word_emb_weight_key in sharded_state_dict
    del sharded_state_dict[word_emb_weight_key]
    sharded_state_dict[word_emb_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=word_emb_weight,
        key=word_emb_weight_key,
        replica_id=mtp_word_emb_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict,
    output_layer_weight: Tensor,
    output_layer_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """Tie the output-layer weight of the MTP stage in a sharded state dict."""
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post-processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert output_layer_weight_key in sharded_state_dict
    del sharded_state_dict[output_layer_weight_key]
    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=output_layer_weight,
        key=output_layer_weight_key,
        replica_id=mtp_output_layer_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


# ---------------------------------------------------------------------------
# roll_tensor — Context-Parallel-aware sequence rolling
# ---------------------------------------------------------------------------

def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    """Roll the tensor along the sequence dimension with CP support.

    For CP=1 (default): standard ``torch.roll`` with zero padding at the
    boundary position.
    For CP>1: splits tensor into chunks, rolls within each chunk, then
    exchanges boundary elements between adjacent CP ranks.
    For packed sequences: respects sequence boundaries.

    Args:
        tensor:           Input tensor to roll, or ``None`` (returns ``(None, None)``).
        shifts:           Shift amount (typically -1 for MTP).
        dims:             Dimension to roll (typically -1 for the sequence dim).
        cp_group:         Context-parallelism process group.
        packed_seq_params: If provided, per-sequence rolling respecting boundaries.

    Returns:
        Tuple[Tensor, Tensor]: (rolled_tensor, sum_of_rolled_tensor)
    """
    if tensor is None:
        return None, None

    if packed_seq_params is not None:
        return _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group)

    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
        rolled_tensor.select(dims, shifts).fill_(0)
        return rolled_tensor, rolled_tensor.sum()

    # CP-enabled rolling.
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = [torch.roll(t, shifts=shifts, dims=dims) for t in tensor_list]

    tensor_send_list = []
    tensor_recv_list = []
    for rt in rolled_tensor_list:
        tensor_send_list.append(rt.select(dims, shifts).contiguous())
        tensor_recv_list.append(
            torch.empty(tensor_send_list[-1].shape,
                        dtype=tensor_send_list[-1].dtype,
                        device=torch.cuda.current_device())
        )

    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    ops = []
    if local_rank != 0:
        ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
        ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
    else:
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
        ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
    else:
        tensor_recv_list[0] = tensor_send_list[1]

    for op in ops:
        op.wait()

    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i, rt in enumerate(rolled_tensor_list):
        rt[tuple(index)] = tensor_recv_list[i]

    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)
    return rolled_tensor, rolled_tensor.sum()


def _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Roll tensor with packed-sequence support, respecting sequence boundaries."""
    assert (
        dims == -1 or dims == tensor.dim() - 1
    ), "Packed sequence roll only supports the last dimension."
    assert shifts == -1, "Packed sequence roll only supports a single-token left shift."
    cu_seqlens = packed_seq_params.cu_seqlens_q
    assert cu_seqlens is not None, "Packed sequence parameters must provide cu_seqlens_q."

    rolled_tensor = tensor.clone()

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_slice = tensor[..., start_idx:end_idx]
            rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
            rolled_seq[..., shifts:] = 0
            rolled_tensor[..., start_idx:end_idx] = rolled_seq
        return rolled_tensor, rolled_tensor.sum()

    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]
        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size
        local_seq_len = local_end_idx - local_start_idx
        if local_seq_len == 0:
            continue

        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            if chunk.size(dims) == 0:
                tensor_send_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                tensor_recv_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                continue
            boundary = chunk.select(dims, shifts).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        ops = []
        if local_rank != 0:
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
        else:
            tensor_recv_list[1].zero_()

        if local_rank != cp_size - 1:
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
        else:
            tensor_recv_list[0].copy_(tensor_send_list[1])

        for op in ops:
            op.wait()

        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = shifts
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            if chunk.size(dims) == 0:
                continue
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    return rolled_tensor, rolled_tensor.sum()


# ---------------------------------------------------------------------------
# MTPLossLoggingHelper
# ---------------------------------------------------------------------------

class MTPLossLoggingHelper:
    """Helper class for logging MTP losses and acceptance rates."""

    tracker: dict = {}

    @staticmethod
    def save_metrics_to_tracker(
        loss: torch.Tensor,
        correct: torch.Tensor,
        total: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save MTP metrics (loss, correct, total) for later logging."""
        if layer_number is None:
            return
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            tracker["loss_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        if "correct_values" not in tracker:
            tracker["correct_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        if "total_values" not in tracker:
            tracker["total_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["loss_values"][layer_number] += loss.detach()
        tracker["correct_values"][layer_number] += correct.detach()
        tracker["total_values"][layer_number] += total.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_metrics_in_tracker():
        """Clear MTP metrics in the tracker."""
        tracker = MTPLossLoggingHelper.tracker
        for key in ("loss_values", "correct_values", "total_values"):
            if key in tracker:
                tracker[key].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_metrics_in_tracker():
        """Collect and reduce MTP metrics across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            return
        loss_values = tracker["loss_values"]
        if tracker.get("reduce_group") is not None:
            torch.distributed.all_reduce(loss_values, group=tracker["reduce_group"])
        if tracker.get("avg_group") is not None:
            torch.distributed.all_reduce(
                loss_values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG
            )
        for key in ("correct_values", "total_values"):
            if key not in tracker:
                continue
            values = tracker[key]
            if tracker.get("reduce_group") is not None:
                torch.distributed.all_reduce(values, group=tracker["reduce_group"])
            if tracker.get("avg_group") is not None:
                torch.distributed.all_reduce(
                    values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.SUM
                )

    @staticmethod
    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track MTP metrics for logging (TensorBoard / WandB)."""
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            return

        mtp_losses = tracker["loss_values"] * loss_scale
        mtp_corrects = tracker.get("correct_values", torch.zeros_like(mtp_losses))
        mtp_totals = tracker.get("total_values", torch.ones_like(mtp_losses))

        if (
            "cumulative_correct_values" not in tracker
            or tracker["cumulative_correct_values"].shape != mtp_corrects.shape
        ):
            tracker["cumulative_correct_values"] = torch.zeros_like(mtp_corrects)
        if (
            "cumulative_total_values" not in tracker
            or tracker["cumulative_total_values"].shape != mtp_totals.shape
        ):
            tracker["cumulative_total_values"] = torch.zeros_like(mtp_totals)

        tracker["cumulative_correct_values"] += mtp_corrects
        tracker["cumulative_total_values"] += mtp_totals
        mtp_cumulative_corrects = tracker["cumulative_correct_values"]
        mtp_cumulative_totals = tracker["cumulative_total_values"]

        for i in range(mtp_losses.shape[0]):
            loss_name = f"mtp_{i + 1} loss"
            step_acc_name = f"mtp_{i + 1}_acceptance_rate"
            cum_acc_name = f"mtp_{i + 1}_cumulative_acceptance_rate"

            loss = mtp_losses[i]
            step_rate = (mtp_corrects[i] / torch.clamp(mtp_totals[i], min=1)) * 100.0
            cum_rate = (
                mtp_cumulative_corrects[i] / torch.clamp(mtp_cumulative_totals[i], min=1)
            ) * 100.0

            if total_loss_dict is not None:
                total_loss_dict[loss_name] = (
                    total_loss_dict.get(loss_name, torch.zeros_like(loss)) + loss
                )
            if writer is not None:
                writer.add_scalar(loss_name, loss, iteration)
                writer.add_scalar(step_acc_name, step_rate, iteration)
                writer.add_scalar(cum_acc_name, cum_rate, iteration)
            if wandb_writer is not None:
                wandb_writer.log({loss_name: loss}, iteration)
                wandb_writer.log({step_acc_name: step_rate}, iteration)
                wandb_writer.log({cum_acc_name: cum_rate}, iteration)

        MTPLossLoggingHelper.clean_metrics_in_tracker()


# ---------------------------------------------------------------------------
# MTP acceptance-count helpers
# ---------------------------------------------------------------------------

def _mtp_logits_are_vocab_sharded(
    output_layer: Callable, runtime_gather_output: Optional[bool]
) -> bool:
    """Return whether MTP logits are vocab-sharded across tensor-parallel ranks."""
    if runtime_gather_output is not None:
        return not runtime_gather_output
    return not getattr(output_layer, "gather_output", False)


def _vocab_parallel_argmax(
    vocab_parallel_logits: Tensor,
    tp_group: torch.distributed.ProcessGroup,
    tp_size: int,
) -> Tensor:
    """Return global argmax IDs from logits sharded across the vocab dimension."""
    vocab_shard_size = vocab_parallel_logits.size(-1)
    local_max_vals, local_argmax = vocab_parallel_logits.max(dim=-1)

    gathered_max_vals = [torch.empty_like(local_max_vals) for _ in range(tp_size)]
    gathered_argmax = [torch.empty_like(local_argmax) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered_max_vals, local_max_vals, group=tp_group)
    torch.distributed.all_gather(gathered_argmax, local_argmax, group=tp_group)

    stacked_max_vals = torch.stack(gathered_max_vals, dim=0)
    stacked_argmax = torch.stack(gathered_argmax, dim=0)
    winning_rank = stacked_max_vals.argmax(dim=0)
    winning_local_argmax = torch.gather(
        stacked_argmax, 0, winning_rank.unsqueeze(0)
    ).squeeze(0)
    return winning_rank * vocab_shard_size + winning_local_argmax


def _compute_mtp_acceptance_counts(
    mtp_logits: Tensor,
    mtp_labels: Tensor,
    loss_mask: Tensor,
    output_layer: Callable,
    runtime_gather_output: Optional[bool],
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple:
    """Compute MTP acceptance correct/total counts."""
    with torch.no_grad():
        logits_are_vocab_sharded = _mtp_logits_are_vocab_sharded(
            output_layer, runtime_gather_output
        )
        if (
            tp_group is None
            and logits_are_vocab_sharded
            and parallel_state.is_initialized()
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        ):
            raise ValueError(
                "tp_group must be provided when computing MTP acceptance counts "
                "from vocab-sharded logits under tensor model parallelism."
            )
        tp_size = torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1

        if tp_group is not None and tp_size > 1 and logits_are_vocab_sharded:
            preds = _vocab_parallel_argmax(mtp_logits, tp_group, tp_size)
        else:
            preds = torch.argmax(mtp_logits, dim=-1)

        labels_match = mtp_labels.transpose(0, 1).contiguous()
        mask_match = loss_mask.transpose(0, 1).contiguous()
        valid_positions = mask_match.bool()
        correct = ((preds == labels_match) & valid_positions).sum().float()
        total = valid_positions.sum().float()

    return correct, total


# ---------------------------------------------------------------------------
# MultiTokenPredictionLayerSubmodules
# ---------------------------------------------------------------------------

@dataclass
class MultiTokenPredictionLayerSubmodules:
    """Submodule spec for a single MultiTokenPrediction layer.

    Args:
        enorm: Embedding normalisation builder.
        hnorm: Hidden-state normalisation builder.
        layer_norm: Final layer-norm builder.
        eh_proj: Linear projection spec (2H → H).
        mtp_model_layer: Inner transformer / Mamba layer spec.
    """
    enorm: LayerNormBuilder
    hnorm: LayerNormBuilder
    layer_norm: LayerNormBuilder
    eh_proj: Union[ModuleSpec, type] = None
    mtp_model_layer: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(mtp_model_layer_spec, use_transformer_engine: bool):
    """Return an MTP layer ModuleSpec using either TE or local backend."""
    try:
        if use_transformer_engine:
            from deepspeed.core.models.backends import TESpecProvider
            backend = TESpecProvider()
        else:
            from deepspeed.core.models.backends import LocalSpecProvider
            backend = LocalSpecProvider()
        return get_mtp_layer_spec_for_backend(mtp_model_layer_spec, backend)
    except ImportError:
        # Fall back to a generic ModuleSpec using default layer norm.
        return get_mtp_layer_spec_for_backend(mtp_model_layer_spec, backend=None)


def get_mtp_layer_spec_for_backend(mtp_model_layer_spec, backend=None):
    """Return an MTP layer ModuleSpec using modules from *backend*.

    When *backend* is None (DS port default), uses ``torch.nn.LayerNorm`` and
    ``torch.nn.Linear`` as column-parallel placeholders.
    """
    import torch.nn as nn
    if backend is not None:
        column_parallel_linear_impl = backend.column_parallel_linear()
        layer_norm_impl = backend.layer_norm()
    else:
        column_parallel_linear_impl = nn.Linear
        layer_norm_impl = nn.LayerNorm

    return ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            mtp_model_layer=mtp_model_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline-parallel layout helpers
# ---------------------------------------------------------------------------

def mtp_on_this_rank(
    layout=None,
    mtp_num_layers: Optional[int] = None,
    ignore_virtual: Optional[bool] = True,
    vp_stage: Optional[int] = None,
) -> bool:
    """Return True if MTP layers should run on the current rank."""
    _mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if layout is not None and HAS_PP_LAYOUT:
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout.layout[pp_rank][vp_stage].count(LayerType.mtp)
            _mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout.layout[pp_rank])):
                if layout.layout[pp_rank][vpp_rank].count(LayerType.mtp) > 0:
                    _mtp_on_this_rank = True
                    break
    else:
        if mtp_num_layers is not None:
            _mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                ignore_virtual=ignore_virtual, vp_stage=vp_stage
            )
    return _mtp_on_this_rank


def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Return the global ranks that host MTP layers."""
    if config.mtp_num_layers is None:
        return []
    if not HAS_PP_LAYOUT or config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    mtp_ranks: set = set()
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank_idx, pp_rank in enumerate(pp_ranks):
        for vpp_rank in range(len(layout[pp_rank_idx])):
            if layout[pp_rank_idx][vpp_rank].count(LayerType.mtp):
                mtp_ranks.add(pp_rank)
    return list(mtp_ranks)


def get_mtp_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Return the MTP layer offset for pipeline-parallel staging."""
    if config.pipeline_model_parallel_size > 1:
        if HAS_PP_LAYOUT and getattr(config, "pipeline_model_parallel_layout", None):
            return config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.mtp, vp_stage=vp_stage
            )
    return 0


def get_mtp_num_layers_to_build(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> int:
    """Return the number of MTP layers to build on the current rank."""
    if HAS_PP_LAYOUT and getattr(config, "pipeline_model_parallel_layout", None) is not None:
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, all MTP layers must be on the last pipeline stage, "
            f"so the count ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
        return num_layers_to_build
    if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
        return config.mtp_num_layers if config.mtp_num_layers else 0
    return 0


# ---------------------------------------------------------------------------
# MTPLossAutoScaler
# ---------------------------------------------------------------------------

class MTPLossAutoScaler(torch.autograd.Function):
    """AutoScaler that triggers the MTP loss backward and scales its gradient."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    # From Megatron M3312 (PR #3159): track original_num_tokens (before any
    # MTP roll) so that _get_mtp_loss_scale in schedules.py can apply the
    # correct token-ratio correction when calculate_per_token_loss=True.
    # Set by process_mtp_loss() at the start of each forward pass; read by
    # _get_mtp_loss_scale() to fold the ratio into the AutoScaler scale.
    original_num_tokens: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mtp_loss,) = ctx.saved_tensors
        scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the backward scale for the MTP loss."""
        MTPLossAutoScaler.main_loss_backward_scale = scale

    @staticmethod
    def set_original_num_tokens(num_tokens: Optional[torch.Tensor]) -> None:
        """Record the pre-roll token count for use by the schedule-level scaler.

        Called once per forward pass by ``process_mtp_loss`` before the first
        roll, so that ``_get_mtp_loss_scale`` in
        ``deepspeed.core.pipeline_parallel.schedules`` can read it back and
        apply the ``original_num_tokens / rolled_num_tokens`` correction factor
        required by Megatron M3312 (PR #3159) when
        ``config.calculate_per_token_loss=True``.
        """
        MTPLossAutoScaler.original_num_tokens = num_tokens


# ---------------------------------------------------------------------------
# process_mtp_loss
# ---------------------------------------------------------------------------

def process_mtp_loss(
    hidden_states: Tensor,
    labels: Tensor,
    loss_mask: Optional[Tensor],
    output_layer: Callable,
    output_weight: Optional[Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params=None,
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
    input_ids: Optional[Tensor] = None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss.

    Splits concatenated hidden states and computes MTP losses for each
    prediction depth.  Used on the post-process rank.

    Args:
        hidden_states: Concatenated hidden states ``[s*(1+mtp_num_layers), b, h]``.
        labels:        Ground-truth labels, or ``None`` when derived from *input_ids*.
        loss_mask:     Boolean mask; all-ones when ``None``.
        output_layer:  Callable to project hidden → logits.
        output_weight: Optional weight for tied embeddings.
        runtime_gather_output: Whether logits are gathered at runtime.
        is_training:   Training flag for metrics logging.
        compute_language_model_loss: Cross-entropy function.
        config:        TransformerConfig with mtp_num_layers, etc.
        cp_group:      Context-parallelism process group.
        tp_group:      Tensor-parallelism process group.
        packed_seq_params: Packed-sequence parameters.
        scale_logits_fn: Optional logit scaling (μP).
        input_ids:     Token IDs used to derive labels when labels is None.

    Returns:
        Updated *hidden_states* (first chunk only, shape ``[s, b, h]``).
    """
    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    derived_labels_from_input_ids = False
    if labels is None:
        if input_ids is None:
            return hidden_states
        labels, _ = roll_tensor(
            input_ids, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        derived_labels_from_input_ids = True

    if config.mtp_detach_heads:
        if output_weight is not None:
            output_weight = output_weight.detach()
        else:
            output_weight = output_layer.weight.detach()

    mtp_labels = labels.clone()
    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)
    if derived_labels_from_input_ids:
        loss_mask, _ = roll_tensor(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )

    original_num_tokens = loss_mask.sum()
    # M3312: publish the pre-roll token count so schedules._get_mtp_loss_scale
    # can apply the original/rolled ratio correction when
    # calculate_per_token_loss=True.
    MTPLossAutoScaler.set_original_num_tokens(original_num_tokens)

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            mtp_logits = scale_logits_fn(mtp_logits)
        mtp_labels, _ = roll_tensor(
            mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        loss_mask, num_tokens = roll_tensor(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )

        mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
        mtp_loss = loss_mask * mtp_loss

        if is_training:
            mtp_loss_for_log = (
                torch.sum(mtp_loss) * (num_tokens > 0).to(mtp_loss.dtype)
            ) / num_tokens.clamp(min=1)
            correct, total = _compute_mtp_acceptance_counts(
                mtp_logits, mtp_labels, loss_mask, output_layer, runtime_gather_output, tp_group
            )
            MTPLossLoggingHelper.save_metrics_to_tracker(
                mtp_loss_for_log,
                correct,
                total,
                mtp_layer_number,
                config.mtp_num_layers,
                avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )

        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            num_tokens_safe = torch.clamp(num_tokens, min=1)
            mtp_loss_normalized = (
                mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
            )
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
        else:
            safe_num_tokens = num_tokens.clamp(min=1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens
            )

    return hidden_states


# ---------------------------------------------------------------------------
# MultiTokenPredictionLayer
# ---------------------------------------------------------------------------

class MultiTokenPredictionLayer(MegatronModule):
    """Single Multi-Token Prediction (MTP) depth module.

    Implements one prediction-depth of MTP (DeepSeek-V3, §3):

        1. Roll input_ids / position_ids left by 1 to obtain the *next* token.
        2. Embed the next token → ``decoder_input``.
        3. Normalise ``hidden_states`` (hnorm) and ``decoder_input`` (enorm).
        4. Concatenate → linear projection 2H→H (``eh_proj``).
        5. Forward through the inner transformer / Mamba layer.
        6. Apply ``final_layernorm`` and return.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mtp_layer_pattern: Optional[str] = None,
        hybrid_submodules=None,
        mamba_submodules=None,
        name: Optional[str] = None,
    ):
        super().__init__(config=config)
        if mamba_submodules is not None:
            if hybrid_submodules is not None:
                raise ValueError(
                    "Cannot specify both hybrid_submodules and mamba_submodules. "
                    "mamba_submodules has been deprecated; use hybrid_submodules instead."
                )
            warnings.warn(
                "mamba_submodules has been deprecated. Use hybrid_submodules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hybrid_submodules = mamba_submodules

        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number + get_mtp_layer_offset(config, vp_stage)
        self.vp_stage = vp_stage
        self.cp_group = pg_collection.cp if pg_collection is not None else None
        self.tp_group = pg_collection.tp if pg_collection is not None else None
        self.mtp_layer_pattern = mtp_layer_pattern

        # ---- enorm / hnorm ------------------------------------------------
        self.enorm = submodules.enorm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.hnorm = submodules.hnorm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )

        # ---- eh_proj (2H → H) ---------------------------------------------
        self.eh_proj = build_module(
            submodules.eh_proj,
            config.hidden_size * 2,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="mtp_eh_proj",
            tp_group=self.tp_group,
            name=(name + ".eh_proj") if name is not None else None,
        )

        # ---- inner transformer / Mamba layer --------------------------------
        if mtp_layer_pattern is not None and hybrid_submodules is not None and HAS_HYBRID:
            from deepspeed.core.models.hybrid.hybrid_block import HybridStack
            from deepspeed.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
            self.mtp_model_layer = HybridStack(
                config=config,
                submodules=hybrid_submodules,
                layer_type_list=validate_segment_layers(mtp_layer_pattern),
                pp_layer_offset=0,
                pre_process=True,
                post_layer_norm=False,
                post_process=True,
                pg_collection=pg_collection,
                is_mtp_layer=True,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )
        elif config.mtp_num_layers is not None:
            self.mtp_model_layer = build_module(
                submodules.mtp_model_layer,
                config=config,
                vp_stage=vp_stage,
                layer_number=self.layer_number,
                is_mtp_layer=True,
                pg_collection=pg_collection,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )

        # ---- final_layernorm -----------------------------------------------
        self.final_layernorm = submodules.layer_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.offload_context = nullcontext()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Roll ids, embed the next token, and prepare hidden states."""
        input_ids, _ = roll_tensor(
            input_ids, shifts=-1, dims=-1,
            cp_group=self.cp_group, packed_seq_params=packed_seq_params,
        )
        position_ids, _ = roll_tensor(
            position_ids, shifts=-1, dims=-1,
            cp_group=self.cp_group, packed_seq_params=packed_seq_params,
        )
        if padding_mask is not None:
            padding_mask, _ = roll_tensor(
                padding_mask, shifts=-1, dims=-1,
                cp_group=self.cp_group, packed_seq_params=packed_seq_params,
            )
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        if self.config.mtp_detach_heads:
            decoder_input = decoder_input.detach()

        hidden_states = _make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        if not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        return input_ids, position_ids, padding_mask, decoder_input, hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """Normalise, concatenate, and project embeddings + hidden states."""
        decoder_input = apply_module(self.enorm)(decoder_input)
        decoder_input = _make_viewless_tensor(decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = apply_module(self.hnorm)(hidden_states)
        hidden_states = _make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)

        # Gather across TP ranks after the column-parallel projection.
        # (DS port: inference path also uses the standard gather.)
        hidden_states = gather_from_tensor_model_parallel_region(
            hidden_states, group=self.tp_group
        )
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_group
            )
        return hidden_states

    def _postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the final layer norm and return a viewless tensor."""
        hidden_states = apply_module(self.final_layernorm)(hidden_states)
        hidden_states = _make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate embeddings with hidden states then apply transformer forward."""
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # FP8 context — falls back to nullcontext when not available.
        fp8_context = get_fp8_context(self.config)
        transformer_layer_fp8_context = get_fp8_context(self.config)

        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)
            with transformer_layer_fp8_context:
                if self.mtp_layer_pattern is not None:
                    hidden_states = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_context=inference_params,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    hidden_states, _ = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                    )

        hidden_states = self._postprocess(hidden_states)
        return hidden_states

    # ------------------------------------------------------------------
    # Speculative-decoding forward (no rolling)
    # ------------------------------------------------------------------

    def forward_single_position(
        self,
        hidden_states: Tensor,
        next_token_ids: Tensor,
        position_ids: Tensor,
        embedding: Callable,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward for single positions without rolling (speculative decoding)."""
        decoder_input = embedding(input_ids=next_token_ids, position_ids=position_ids)
        hidden_states = _make_viewless_tensor(hidden_states, requires_grad=False, keep_graph=False)
        hidden_states = self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return hidden_states

    # ------------------------------------------------------------------
    # Activation-recompute forward
    # ------------------------------------------------------------------

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """Forward through ``_proj_and_transformer_layer`` with activation recompute."""

        def custom_forward(
            hidden_states, decoder_input, attention_mask, padding_mask,
            context, context_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
            sequence_len_offset,
        ):
            return self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        if self.config.recompute_method == "uniform":
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            outputs = tensor_parallel.checkpoint(
                custom_forward,
                self.config.distribute_saved_activations,
                hidden_states,
                decoder_input,
                attention_mask,
                padding_mask,
                context,
                context_mask,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        elif self.config.recompute_method == "block":
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet. Skipping recompute."
            )
            outputs = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            raise ValueError("Invalid activation recompute method.")
        return outputs

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
        embedding=None,
    ):
        """Execute one MTP depth forward pass.

        Args:
            input_ids:       Token IDs ``[b, s]``.
            position_ids:    Position IDs ``[b, s]``.
            hidden_states:   Decoder hidden states ``[s, b, h]``.
            attention_mask:  Self-attention mask ``[1, 1, s, s]``.
            embedding:       Embedding module from the GPT model.

        Returns:
            Tuple ``(hidden_states, input_ids, position_ids, padding_mask)`` where
            ``hidden_states`` has shape ``[s, b, h]``.
        """
        assert context is None, "Multi-Token Prediction + cross-attention is not yet supported."
        input_ids, position_ids, padding_mask, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            padding_mask=padding_mask,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        if self.config.recompute_granularity == "full" and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        return hidden_states, input_ids, position_ids, padding_mask

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        # Backward-compat: old GPT-MTP checkpoints used 'transformer_layer'.
        if self.mtp_layer_pattern is None:
            apply_prefix_mapping(
                sharded_state_dict,
                {f"{prefix}mtp_model_layer.": f"{prefix}transformer_layer."},
            )
        return sharded_state_dict


# ---------------------------------------------------------------------------
# MultiTokenPredictionBlockSubmodules
# ---------------------------------------------------------------------------

@dataclass
class MultiTokenPredictionBlockSubmodules:
    """Submodule spec for a full MTP block (all depths).

    Args:
        layer_specs: One ModuleSpec per MTP depth (or a single shared spec).
    """
    layer_specs: Optional[List[ModuleSpec]] = None


def _get_mtp_block_submodules(
    config: TransformerConfig,
    spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec],
) -> MultiTokenPredictionBlockSubmodules:
    """Return ``MultiTokenPredictionBlockSubmodules`` from *spec*."""
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")


# ---------------------------------------------------------------------------
# MultiTokenPredictionBlock
# ---------------------------------------------------------------------------

class MultiTokenPredictionBlock(MegatronModule):
    """Stack of :class:`MultiTokenPredictionLayer` modules.

    Hosts *D* sequential MTP modules (or 1 shared module when
    ``config.mtp_use_repeated_layer=True``).

    Each depth takes the output hidden states from the previous depth, rolls
    the token stream, embeds the next token, projects and attends, and appends
    its output to ``hidden_states_list``.  The concatenated list is returned as
    a single tensor so the post-process rank can split it and compute per-depth
    losses via :func:`process_mtp_loss`.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mtp_layer_pattern: Optional[str] = None,
        mtp_num_depths: int = 0,
        hybrid_submodules=None,
        mamba_submodules=None,
        name: Optional[str] = None,
    ):
        super().__init__(config=config)
        if mamba_submodules is not None:
            if hybrid_submodules is not None:
                raise ValueError(
                    "Cannot specify both hybrid_submodules and mamba_submodules. "
                    "mamba_submodules has been deprecated; use hybrid_submodules instead."
                )
            warnings.warn(
                "mamba_submodules has been deprecated. Use hybrid_submodules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hybrid_submodules = mamba_submodules

        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.vp_stage = vp_stage
        self.mtp_layer_pattern = mtp_layer_pattern
        self.mtp_num_depths = mtp_num_depths
        self.hybrid_submodules = hybrid_submodules
        self.mtp_use_repeated_layer = config.mtp_use_repeated_layer
        self.name = name

        vp_size = config.virtual_pipeline_model_parallel_size
        assert is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size), (
            f"MTP layers must be placed on the last virtual pipeline stage. "
            f"Got vp_stage={vp_stage} with vp_size={vp_size}."
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["cp", "tp"]
            )
        else:
            assert hasattr(pg_collection, "cp"), (
                "MultiTokenPredictionBlock pg_collection must have cp process group"
            )

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."
        self.cp_group = pg_collection.cp

        if config.mtp_detach_heads:
            for param in self.parameters():
                param.grad_norm_group = "mtp"

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self, pg_collection):
        if self.mtp_num_depths > 0:
            num_depths = self.mtp_num_depths
        else:
            num_depths = self.config.mtp_num_layers or len(self.submodules.layer_specs)

        def _build_legacy(layer_spec, layer_number):
            fp8_ctx = get_fp8_context(self.config, is_init=True)
            with fp8_ctx:
                return build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    name=(self.name + f".layers.{layer_number}") if self.name else None,
                )

        def _build_with_pattern(layer_spec, layer_number, pattern, hs):
            fp8_ctx = get_fp8_context(self.config, is_init=True)
            with fp8_ctx:
                return build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=pattern,
                    hybrid_submodules=hs,
                    name=(self.name + f".layers.{layer_number}") if self.name else None,
                )

        pattern = self.mtp_layer_pattern
        hs = self.hybrid_submodules

        if pattern is not None and hs is not None and HAS_HYBRID:
            if self.mtp_use_repeated_layer:
                shared = _build_with_pattern(self.submodules.layer_specs[0], 1, pattern, hs)
                self.layers = torch.nn.ModuleList([shared])
            else:
                self.layers = torch.nn.ModuleList([
                    _build_with_pattern(
                        self.submodules.layer_specs[min(i, len(self.submodules.layer_specs) - 1)],
                        i + 1, pattern, hs,
                    )
                    for i in range(num_depths)
                ])
        elif self.mtp_use_repeated_layer:
            if len(self.submodules.layer_specs) != 1:
                warnings.warn(
                    "Repeated MTP mode expects exactly 1 layer spec, got "
                    f"{len(self.submodules.layer_specs)} instead. "
                    f"The first layer will be applied {self.config.mtp_num_layers} times."
                )
            self.layers = torch.nn.ModuleList(
                [_build_legacy(self.submodules.layer_specs[0], 1)]
            )
        else:
            self.layers = torch.nn.ModuleList([
                _build_legacy(spec, i + 1)
                for i, spec in enumerate(self.submodules.layer_specs)
            ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding=None,
    ) -> Tensor:
        """Forward through all MTP depths.

        Returns:
            Tensor of shape ``[s*(1+mtp_num_layers), b, h]`` — the concatenation
            of the decoder hidden states and all per-depth MTP hidden states.
            :func:`process_mtp_loss` splits this on the post-process rank.
        """
        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states = hidden_states_list[offset]

        if self.config.mtp_detach_heads:
            hidden_states = hidden_states.detach()

        for iteration in range(self.config.mtp_num_layers):
            layer_idx = 0 if self.mtp_use_repeated_layer else iteration
            (hidden_states, input_ids, position_ids, padding_mask) = self.layers[layer_idx](
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=embedding,
                **(extra_block_kwargs or {}),
            )
            hidden_states_list.append(hidden_states)

        return torch.cat(hidden_states_list, dim=0)

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict: ShardedStateDict = {}
        layer_prefix = f"{prefix}layers."
        for layer in self.layers:
            offset = get_mtp_layer_offset(self.config, self.vp_stage)
            sharded_prefix = f"{layer_prefix}{layer.layer_number - 1}."
            state_dict_prefix = f"{layer_prefix}{layer.layer_number - 1 - offset}."
            layer_sharded_sd = layer.sharded_state_dict(state_dict_prefix, (), metadata)
            replace_prefix_for_sharding(layer_sharded_sd, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_sd)
        return sharded_state_dict
