# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Multi-Token Prediction (MTP) — ported from Megatron-LM.

Source: megatron/core/transformer/multi_token_prediction.py
Ported to: deepspeed/core/transformer/multi_token_prediction.py

Import-mapping summary
----------------------
  megatron.core                                 → deepspeed.core
  megatron.core.InferenceParams                 → typing.Any  (shim)
  megatron.core.parallel_state                  → deepspeed.core.parallel_state
  megatron.core.tensor_parallel                 → deepspeed.core.tensor_parallel
  megatron.core.tensor_parallel.{gather,scatter}→ deepspeed.core.tensor_parallel.mappings
  megatron.core.dist_checkpointing.mapping      → deepspeed.core.dist_checkpointing.mapping
  megatron.core.dist_checkpointing.utils        → inline shims (apply_prefix_mapping /
                                                   replace_prefix_for_sharding)
  megatron.core.enums (Fp8Recipe)               → local enum shim
  megatron.core.extensions.transformer_engine   → HAVE_TE = False  (not in DS)
  megatron.core.fp8_utils.get_fp8_context       → local nullcontext shim
  megatron.core.inference.utils.InferenceMode   → local shim
  megatron.core.models.backends (Backend…)      → local shims
  megatron.core.packed_seq_params               → local dataclass shim
  megatron.core.pipeline_parallel.utils         → local is_vp_last_stage shim
  megatron.core.process_groups_config           → deepspeed.core.process_groups_config
  megatron.core.transformer.enums (AttnMaskType)→ local enum shim
  megatron.core.transformer.module              → deepspeed.core.transformer.module
  megatron.core.transformer.spec_utils          → local ModuleSpec / build_module shims
  megatron.core.transformer.torch_norm          → local LayerNormBuilder shim
  megatron.core.transformer.transformer_block   → local TransformerBlockSubmodules shim
  megatron.core.transformer.transformer_config  → deepspeed.core.transformer.transformer_config
  megatron.core.typed_torch.apply_module        → local identity shim
  megatron.core.utils.*                         → local shims (get_pg_rank,
                                                   is_torch_min_version, make_viewless_tensor,
                                                   make_tp_sharded_tensor_for_checkpoint)
  megatron.core.pipeline_parallel_layer_layout  → local shim

All public classes and functions are preserved with identical signatures so that
downstream code (model definitions, training loops, checkpointing) needs no changes.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# deepspeed.core imports (drop-in replacements for megatron.core)
# ---------------------------------------------------------------------------
from deepspeed.core import parallel_state
from deepspeed.core.process_groups_config import ProcessGroupCollection
from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.dist_checkpointing.mapping import ShardedStateDict
from deepspeed.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
import deepspeed.core.tensor_parallel as tensor_parallel

# ---------------------------------------------------------------------------
# Shims for Megatron-only APIs not present in deepspeed.core
# ---------------------------------------------------------------------------

# InferenceParams — used only as a type annotation / passed opaquely to inner layers.
InferenceParams = Any

# PackedSeqParams — minimal shim; real usage requires cu_seqlens_q.
@dataclass
class PackedSeqParams:
    cu_seqlens_q: Optional[Tensor] = None
    cu_seqlens_kv: Optional[Tensor] = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_kv: Optional[int] = None
    qkv_format: Optional[str] = None


# Fp8Recipe enum shim.
class Fp8Recipe(Enum):
    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8_block_scaling = "mxfp8_block_scaling"


# AttnMaskType enum shim.
class AttnMaskType(Enum):
    padding = "padding"
    causal = "causal"
    no_mask = "no_mask"
    padding_causal = "padding_causal"


# HAVE_TE — TransformerEngine not available in the DeepSpeed port.
HAVE_TE = False

# TESpecProvider shim.
TESpecProvider = None


# get_fp8_context shim — always returns nullcontext.
def get_fp8_context(config: TransformerConfig, is_init: bool = False):  # noqa: ANN201
    return nullcontext()


# InferenceMode shim.
class InferenceMode:
    _active: bool = False

    @staticmethod
    def is_active() -> bool:
        return InferenceMode._active


# inference_all_gather_from_tensor_model_parallel_region shim —
# falls back to the standard gather (adequate for non-inference or single-rank usage).
def inference_all_gather_from_tensor_model_parallel_region(
    hidden_states: Tensor,
    tp_group,
    config: TransformerConfig,
) -> Tensor:
    return gather_from_tensor_model_parallel_region(hidden_states)


# apply_module — identity wrapper; in Megatron it handles TE fusion.
def apply_module(module):  # noqa: ANN201
    return module


# make_viewless_tensor shim — returns the tensor unchanged (no view stripping needed).
def make_viewless_tensor(
    inp: Tensor,
    requires_grad: bool = True,
    keep_graph: bool = True,
) -> Tensor:
    if inp is None:
        return inp
    if requires_grad and not inp.requires_grad:
        inp = inp.detach().requires_grad_(True)
    return inp


# get_pg_rank shim — thin wrapper around torch.distributed.get_rank.
def get_pg_rank(pg: Optional[torch.distributed.ProcessGroup]) -> int:
    if pg is None:
        return 0
    return torch.distributed.get_rank(group=pg)


# is_torch_min_version shim.
def is_torch_min_version(min_ver: str) -> bool:
    from packaging.version import Version  # type: ignore[import]
    try:
        return Version(torch.__version__) >= Version(min_ver)
    except Exception:
        parts_have = [int(x) for x in torch.__version__.split(".")[:2] if x.isdigit()]
        parts_need = [int(x) for x in min_ver.split(".")[:2] if x.isdigit()]
        return parts_have >= parts_need


# make_tp_sharded_tensor_for_checkpoint shim — stores the tensor with basic metadata.
def make_tp_sharded_tensor_for_checkpoint(
    tensor: Tensor,
    key: str,
    replica_id: tuple,
    allow_shape_mismatch: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> dict:
    return {
        "param": tensor,
        "key": key,
        "replica_id": replica_id,
        "allow_shape_mismatch": allow_shape_mismatch,
    }


# apply_prefix_mapping shim — renames keys in sharded_state_dict in-place.
def apply_prefix_mapping(
    sharded_state_dict: ShardedStateDict,
    mapping: Dict[str, str],
) -> None:
    for old_prefix, new_prefix in mapping.items():
        keys_to_rename = [k for k in list(sharded_state_dict) if k.startswith(old_prefix)]
        for old_key in keys_to_rename:
            new_key = new_prefix + old_key[len(old_prefix):]
            sharded_state_dict[new_key] = sharded_state_dict.pop(old_key)


# replace_prefix_for_sharding shim — renames keys that start with old_prefix.
def replace_prefix_for_sharding(
    sharded_state_dict: ShardedStateDict,
    old_prefix: str,
    new_prefix: str,
) -> None:
    keys_to_rename = [k for k in list(sharded_state_dict) if k.startswith(old_prefix)]
    for old_key in keys_to_rename:
        new_key = new_prefix + old_key[len(old_prefix):]
        sharded_state_dict[new_key] = sharded_state_dict.pop(old_key)


# TransformerBlockSubmodules shim — used only as a type-check target.
@dataclass
class TransformerBlockSubmodules:
    layer_specs: Optional[List] = None


# LayerNormBuilder — callable type alias for norm factories (Protocol in DS).
# Accepted as Union[type, Callable]; no change needed beyond annotation.
LayerNormBuilder = Union[type, Callable]


# ModuleSpec / build_module shims.
@dataclass
class ModuleSpec:
    module: type
    params: dict = None
    submodules: Any = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


def build_module(spec: ModuleSpec, **kwargs) -> nn.Module:
    """Instantiate a module from a ModuleSpec, merging spec.params and kwargs."""
    if isinstance(spec, ModuleSpec):
        merged = {**(spec.params or {}), **kwargs}
        return spec.module(**merged)
    # If spec is already a type, instantiate directly.
    if isinstance(spec, type):
        return spec(**kwargs)
    raise TypeError(f"build_module: unexpected spec type {type(spec)}")


# PipelineParallelLayerLayout shim — minimal surface used in this file.
class PipelineParallelLayerLayout:
    layout: list = None  # list[list[list[LayerType]]]

    def get_layer_offset(self, layer_type, vp_stage=None) -> int:
        return 0

    def get_num_layers_to_build(self, layer_type, vp_stage=None) -> int:
        return 0


# LayerType enum shim (only .mtp needed here).
class LayerType(Enum):
    encoder = "encoder"
    decoder = "decoder"
    retro_encoder = "retro_encoder"
    retro_decoder = "retro_decoder"
    retro_decoder_with_retriever = "retro_decoder_with_retriever"
    mtp = "mtp"


# is_vp_last_stage shim.
def is_vp_last_stage(vp_stage: Optional[int], vp_size: Optional[int]) -> bool:
    if vp_size is None or vp_size == 1:
        return True
    if vp_stage is None:
        return True
    return vp_stage == vp_size - 1


# Backend spec shims (used only by get_mtp_layer_spec helpers).
class BackendSpecProvider:
    def column_parallel_linear(self) -> type:
        raise NotImplementedError

    def layer_norm(self) -> type:
        raise NotImplementedError


class LocalSpecProvider(BackendSpecProvider):
    def column_parallel_linear(self) -> type:
        # Lazy import to avoid circular deps.
        from deepspeed.core.tensor_parallel.layers import ColumnParallelLinear
        return ColumnParallelLinear

    def layer_norm(self) -> type:
        from deepspeed.core.transformer.transformer_layer import _build_norm
        # _build_norm is a factory function, wrap it so it can be called with
        # (config, hidden_size, eps) keyword args.
        class _NormWrapper:
            def __new__(cls, config, hidden_size, eps):
                return _build_norm(config, hidden_size=hidden_size)
        return _NormWrapper


# dist_all_gather_func — use the modern API.
if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base  # type: ignore[attr-defined]

# Supported attention mask types.
SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]

# ---------------------------------------------------------------------------
# Helper functions (unchanged logic, Megatron imports replaced)
# ---------------------------------------------------------------------------


def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict,
    word_emb_weight: Tensor,
    word_emb_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """Tie the embedding of the MTP processing stage in a given sharded state dict.

    Args:
        sharded_state_dict: State dict with the weight to tie.
        word_emb_weight: Weight of the word embedding.
        word_emb_weight_key: Key of the word embedding in the sharded state dict.
        tp_group: The tensor parallel group.
        dp_cp_group: The dp-cp comm group.
    """
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre processing stage
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
    """Tie the output layer of the MTP processing stage in a given sharded state dict.

    Args:
        sharded_state_dict: State dict with the weight to tie.
        output_layer_weight: Weight of the output layer.
        output_layer_weight_key: Key of the output layer in the sharded state dict.
        tp_group: The tensor parallel group.
        dp_cp_group: The dp-cp comm group.
    """
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post processing stage
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


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    """Roll the tensor along the sequence dimension with Context Parallelism (CP) support.

    For CP=1 (default): standard torch.roll with zero padding.
    For CP>1: splits tensor into chunks, rolls within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.
    For packed sequences: respects sequence boundaries when rolling.

    Args:
        tensor: Input tensor to roll. If None, returns (None, None).
        shifts: The shift (typically -1 for MTP).
        dims: The dimension to roll (typically -1 for sequence dimension).
        cp_group: The context parallelism process group.
        packed_seq_params: Parameters for packed sequence processing.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
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
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=torch.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        tensor_recv_list[0] = tensor_send_list[1]

    for op in ops:
        op.wait()

    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)
    return rolled_tensor, rolled_tensor.sum()


def _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Roll tensor with packed sequence support, respecting sequence boundaries."""
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
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """Save the MTP metrics (loss, correct, total) for logging."""
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
    def clean_metrics_in_tracker() -> None:
        """Clear the MTP metrics."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" in tracker:
            tracker["loss_values"].zero_()
        if "correct_values" in tracker:
            tracker["correct_values"].zero_()
        if "total_values" in tracker:
            tracker["total_values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_metrics_in_tracker() -> None:
        """Collect and reduce the MTP metrics across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            return

        loss_values = tracker["loss_values"]
        if tracker.get("reduce_group") is not None:
            torch.distributed.all_reduce(loss_values, group=tracker.get("reduce_group"))
        if tracker.get("avg_group") is not None:
            torch.distributed.all_reduce(
                loss_values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG
            )

        for key in ["correct_values", "total_values"]:
            if key not in tracker:
                continue
            values = tracker[key]
            if tracker.get("reduce_group") is not None:
                torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
            if tracker.get("avg_group") is not None:
                torch.distributed.all_reduce(
                    values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.SUM
                )

    @staticmethod
    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track Multi-Token Prediction (MTP) metrics for logging."""
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

        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            loss_name = f"mtp_{i+1} loss"
            step_acc_name = f"mtp_{i+1}_acceptance_rate"
            cum_acc_name = f"mtp_{i+1}_cumulative_acceptance_rate"

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
                wandb_writer.log({f"{loss_name}": loss}, iteration)
                wandb_writer.log({f"{step_acc_name}": step_rate}, iteration)
                wandb_writer.log({f"{cum_acc_name}": cum_rate}, iteration)

        MTPLossLoggingHelper.clean_metrics_in_tracker()


def _mtp_logits_are_vocab_sharded(
    output_layer: Callable, runtime_gather_output: Optional[bool]
) -> bool:
    """Return whether MTP logits are still vocab-sharded across tensor-parallel ranks."""
    if runtime_gather_output is not None:
        return not runtime_gather_output
    return not getattr(output_layer, "gather_output", False)


def _vocab_parallel_argmax(
    vocab_parallel_logits: Tensor,
    tp_group: torch.distributed.ProcessGroup,
    tp_size: int,
) -> Tensor:
    """Return global argmax ids from logits sharded across the vocab dimension."""
    vocab_shard_size = vocab_parallel_logits.size(-1)
    local_max_vals, local_argmax = vocab_parallel_logits.max(dim=-1)

    gathered_max_vals = [torch.empty_like(local_max_vals) for _ in range(tp_size)]
    gathered_argmax = [torch.empty_like(local_argmax) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered_max_vals, local_max_vals, group=tp_group)
    torch.distributed.all_gather(gathered_argmax, local_argmax, group=tp_group)

    stacked_max_vals = torch.stack(gathered_max_vals, dim=0)
    stacked_argmax = torch.stack(gathered_argmax, dim=0)
    winning_rank = stacked_max_vals.argmax(dim=0)
    winning_local_argmax = torch.gather(stacked_argmax, 0, winning_rank.unsqueeze(0)).squeeze(0)
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
    """Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        enorm: Specification or instance of the embedding normalization.
        hnorm: Specification or instance of the hidden states normalization.
        layer_norm: Specification or instance of the final layer normalization.
        eh_proj: Specification or instance of the linear projection.
        mtp_model_layer: Specification or instance of the transformer block.
    """

    enorm: LayerNormBuilder
    hnorm: LayerNormBuilder
    layer_norm: LayerNormBuilder
    eh_proj: Union[ModuleSpec, type] = None
    mtp_model_layer: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(
    mtp_model_layer_spec: ModuleSpec, use_transformer_engine: bool
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec with Local (or TE) modules.
    """
    return get_mtp_layer_spec_for_backend(
        mtp_model_layer_spec,
        backend=LocalSpecProvider(),
    )


def get_mtp_layer_spec_for_backend(
    mtp_model_layer_spec: ModuleSpec, backend: BackendSpecProvider
) -> ModuleSpec:
    """Get the MTP layer spec for the given backend.

    Returns:
        ModuleSpec with modules from the backend.
    """
    column_parallel_linear_impl = backend.column_parallel_linear()
    layer_norm_impl = backend.layer_norm()
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            mtp_model_layer=mtp_model_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )
    return mtp_layer_spec


# ---------------------------------------------------------------------------
# Pipeline / layout helpers
# ---------------------------------------------------------------------------

def mtp_on_this_rank(
    layout: Optional[PipelineParallelLayerLayout] = None,
    mtp_num_layers: Optional[int] = None,
    ignore_virtual: Optional[bool] = True,
    vp_stage: Optional[int] = None,
) -> bool:
    """Check if there is MTP on the current rank."""
    _mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if layout is not None:
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout.layout[pp_rank][vp_stage].count(LayerType.mtp)
            _mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout.layout[pp_rank])):
                num_layers_to_build = layout.layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    _mtp_on_this_rank = True
                    break
    else:
        if mtp_num_layers is not None:
            _mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                ignore_virtual=ignore_virtual, vp_stage=vp_stage
            )
        else:
            _mtp_on_this_rank = False
    return _mtp_on_this_rank


def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Get the global ranks that hold MTP layers."""
    mtp_ranks: set = set()
    if config.mtp_num_layers is None:
        return []
    if config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank in range(len(layout)):
        for vpp_rank in range(len(layout[pp_rank])):
            num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
            if num_layers_to_build:
                mtp_ranks.add(pp_ranks[pp_rank])
    return list(mtp_ranks)


def get_mtp_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the offset of the MTP layer within the concatenated hidden-states tensor."""
    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.mtp, vp_stage=vp_stage
            )
        else:
            offset = 0
    else:
        offset = 0
    return offset


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build on this rank/stage."""
    if config.pipeline_model_parallel_layout is not None:
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support putting all MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            num_layers_to_build = 0
    return num_layers_to_build


# ---------------------------------------------------------------------------
# MTPLossAutoScaler
# ---------------------------------------------------------------------------

class MTPLossAutoScaler(torch.autograd.Function):
    """AutoScaler that triggers the backward pass and scales the grad for MTP loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        MTPLossAutoScaler.main_loss_backward_scale = scale


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
    packed_seq_params: Optional[PackedSeqParams] = None,
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
    input_ids: Optional[Tensor] = None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss computation.

    This standalone function handles MTP loss computation on the post_process rank.
    It splits concatenated hidden states and computes MTP losses.

    Args:
        hidden_states: Hidden states tensor (concatenated with MTP outputs).
        labels: Ground truth labels.
        loss_mask: Mask for loss computation. If None, uses all ones.
        output_layer: Output layer method to compute logits.
        output_weight: Optional output weight for shared embeddings.
        runtime_gather_output: Whether to gather output at runtime.
        is_training: Whether the model is in training mode.
        compute_language_model_loss: Method to compute language model loss.
        config: Model configuration.
        cp_group: Context parallelism process group.
        tp_group: Tensor parallelism process group.
        packed_seq_params: Packed sequence parameters.
        scale_logits_fn: Optional function to scale logits before loss (e.g. MuP).
        input_ids: Input token IDs; used to derive labels when ``labels`` is None.

    Returns:
        Updated hidden states after MTP loss processing (first chunk only).
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
    """Multi-Token Prediction (MTP) layer — extends prediction to multiple future tokens.

    This MTP implementation sequentially predicts additional tokens and keeps the
    complete causal chain at each prediction depth.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k-1)-th depth, we combine the representation of
    the i-th token and the embedding of the (i+K)-th token with a linear projection.
    The combined representation serves as the input to the Transformer block at depth k.

    Reference: DeepSeek-V3 Technical Report https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mtp_layer_pattern: Optional[str] = None,
        hybrid_submodules: Optional[Any] = None,
        mamba_submodules: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
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

        # Build normalisation layers.
        self.enorm = submodules.enorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.hnorm = submodules.hnorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Linear projection: [s, b, 2*h] → [s, b, h]
        self.eh_proj = build_module(
            submodules.eh_proj,
            config=self.config,
            input_size=self.config.hidden_size * 2,
            output_size=self.config.hidden_size,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="mtp_eh_proj",
            tp_group=self.tp_group,
            name=(name + ".eh_proj") if name is not None else None,
        )

        # Build inner transformer / hybrid layer.
        if mtp_layer_pattern is not None and hybrid_submodules is not None:
            # Hybrid path — lazy import to avoid circular dependency.
            from deepspeed.core.models.hybrid.hybrid_block import HybridStack
            from deepspeed.core.models.hybrid.hybrid_layer_allocation import (
                validate_segment_layers,
            )

            self.mtp_model_layer = HybridStack(
                config=self.config,
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
        elif self.config.mtp_num_layers is not None:
            # GPT path — single TransformerLayer.
            self.mtp_model_layer = build_module(
                submodules.mtp_model_layer,
                config=self.config,
                vp_stage=self.vp_stage,
                layer_number=self.layer_number,
                is_mtp_layer=True,
                pg_collection=pg_collection,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )

        # Final layer norm before shared output head.
        self.final_layernorm = submodules.layer_norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.offload_context = nullcontext()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Preprocess inputs for MTP: roll IDs and compute embedding for next token."""
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

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        if not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        return input_ids, position_ids, padding_mask, decoder_input, hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """Normalise, concatenate, project, and gather embeddings + hidden states."""
        decoder_input = apply_module(self.enorm)(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = apply_module(self.hnorm)(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)

        # Gather across TP ranks after the column-parallel linear projection.
        if InferenceMode.is_active():
            hidden_states = inference_all_gather_from_tensor_model_parallel_region(
                hidden_states, self.tp_group, self.config
            )
        else:
            hidden_states = gather_from_tensor_model_parallel_region(hidden_states)

        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
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
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate embeddings with hidden states, then run transformer layer."""
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

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

        return self._postprocess(hidden_states)

    def _postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer norm and return a viewless tensor."""
        hidden_states = apply_module(self.final_layernorm)(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        return hidden_states

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
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward for single positions without roll_tensor (speculative decoding).

        Unlike the regular forward which rolls input_ids to get the next token's
        embedding, this method directly takes the correct next_token_ids.

        Args:
            hidden_states: Hidden states at positions of interest [N, B, H].
            next_token_ids: The correct next token IDs [B, N].
            position_ids: Position IDs for the next tokens [B, N].
            embedding: The embedding module.

        Returns:
            MTP hidden states [N, B, H].
        """
        decoder_input = embedding(input_ids=next_token_ids, position_ids=position_ids)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=False, keep_graph=False
        )
        return self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

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
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """Forward through _proj_and_transformer_layer with activation recomputation."""

        def custom_forward(
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

        # In DeepSpeed, fp8/fp4 support is shimmed via nullcontext, so we always use
        # tensor_parallel.checkpoint (no te_checkpoint path needed).
        outer_quantization_context = nullcontext()

        def checkpoint_handler():
            return tensor_parallel.checkpoint(
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

        if self.config.recompute_method == "uniform":
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            with outer_quantization_context:
                outputs = checkpoint_handler()
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
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        embedding: Optional[Callable] = None,
    ):
        """Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids: Input token IDs.
            position_ids: Positional IDs.
            hidden_states: [s, b, h] from the preceding layer.
            attention_mask: [1, 1, s, s] self-attention mask.
            padding_mask: Optional padding mask.
            context: Context tensor for cross-attention (not yet supported).
            context_mask: Cross-attention context mask.
            rotary_pos_emb: Rotary positional embeddings.
            rotary_pos_cos: Cosine component of RoPE.
            rotary_pos_sin: Sine component of RoPE.
            attention_bias: Optional attention bias.
            inference_params: Inference cache parameters.
            packed_seq_params: Packed sequence parameters.
            sequence_len_offset: Sequence length offset.
            embedding: The GPT model's embedding callable.

        Returns:
            (hidden_states, input_ids, position_ids, padding_mask)
        """
        assert context is None, "multi token prediction + cross attention is not yet supported."
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

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Generate a sharded state dict for this MTP layer."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # Backward-compat: GPT MTP checkpoints used 'transformer_layer' as the submodule name.
        if self.mtp_layer_pattern is None:
            apply_prefix_mapping(
                sharded_state_dict,
                {f"{prefix}mtp_model_layer.": f"{prefix}transformer_layer."},
            )

        return sharded_state_dict


# ---------------------------------------------------------------------------
# MultiTokenPredictionBlockSubmodules / MultiTokenPredictionBlock
# ---------------------------------------------------------------------------

@dataclass
class MultiTokenPredictionBlockSubmodules:
    """Dataclass for specifying the submodules of a MultiTokenPredictionBlock.

    Args:
        layer_specs: A list of module specifications for the MTP layers.
    """

    layer_specs: Optional[List[ModuleSpec]] = None


def _get_mtp_block_submodules(
    config: TransformerConfig,
    spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec],
) -> MultiTokenPredictionBlockSubmodules:
    """Retrieve or construct MultiTokenPredictionBlockSubmodules from the spec."""
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(MegatronModule):
    """Block of MultiTokenPredictionLayer modules.

    Uses D sequential modules to predict D additional tokens. When
    ``config.mtp_use_repeated_layer`` is True, a single layer is applied
    mtp_num_layers times.

    Reference: DeepSeek-V3 Technical Report https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mtp_layer_pattern: Optional[str] = None,
        mtp_num_depths: int = 0,
        hybrid_submodules: Optional[Any] = None,
        mamba_submodules: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
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
        self.mtp_use_repeated_layer = self.config.mtp_use_repeated_layer
        self.name = name

        vp_size = config.virtual_pipeline_model_parallel_size
        assert is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size), (
            f"MTP layers must be placed on the last virtual pipeline stage. "
            f"Got vp_stage={vp_stage} with vp_size={vp_size}."
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.from_parallel_state()
        else:
            assert hasattr(pg_collection, "cp"), (
                "MultiTokenPredictionBlock pg_collection must have cp process group"
            )

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."
        self.cp_group = pg_collection.cp

        if self.config.mtp_detach_heads:
            for param in self.parameters():
                param.grad_norm_group = "mtp"

    def _build_layers(self, pg_collection: ProcessGroupCollection) -> None:
        num_depths = (
            self.mtp_num_depths
            if self.mtp_num_depths > 0
            else (self.config.mtp_num_layers or len(self.submodules.layer_specs))
        )

        def build_layer_legacy(layer_spec, layer_number):
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    name=(
                        (self.name + f".layers.{layer_number}")
                        if self.name is not None
                        else None
                    ),
                )
            return module

        def build_layer_with_pattern(layer_spec, layer_number, mtp_layer_pattern, hybrid_submodules):
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=mtp_layer_pattern,
                    hybrid_submodules=hybrid_submodules,
                    name=(
                        (self.name + f".layers.{layer_number}")
                        if self.name is not None
                        else None
                    ),
                )
            return module

        if self.mtp_layer_pattern is not None and self.hybrid_submodules is not None:
            if self.mtp_use_repeated_layer:
                layer_spec = self.submodules.layer_specs[0]
                shared_layer = build_layer_with_pattern(
                    layer_spec,
                    layer_number=1,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    hybrid_submodules=self.hybrid_submodules,
                )
                self.layers = torch.nn.ModuleList([shared_layer])
            else:
                self.layers = torch.nn.ModuleList(
                    [
                        build_layer_with_pattern(
                            self.submodules.layer_specs[
                                min(i, len(self.submodules.layer_specs) - 1)
                            ],
                            layer_number=i + 1,
                            mtp_layer_pattern=self.mtp_layer_pattern,
                            hybrid_submodules=self.hybrid_submodules,
                        )
                        for i in range(num_depths)
                    ]
                )
        elif self.mtp_use_repeated_layer:
            if len(self.submodules.layer_specs) != 1:
                warnings.warn(
                    "Repeated MTP mode expects exactly 1 layer spec, got "
                    f"{len(self.submodules.layer_specs)} instead. "
                    f"The first layer will be applied {self.config.mtp_num_layers} times."
                )
            self.layers = torch.nn.ModuleList(
                [build_layer_legacy(self.submodules.layer_specs[0], layer_number=1)]
            )
        else:
            self.layers = torch.nn.ModuleList(
                [
                    build_layer_legacy(layer_spec, i + 1)
                    for i, layer_spec in enumerate(self.submodules.layer_specs)
                ]
            )

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
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding: Optional[Callable] = None,
    ) -> Tensor:
        """Perform the forward pass through all MTP modules.

        Returns:
            Concatenated hidden states tensor [s*(1+num_layers), b, h].
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

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Generate a sharded state dict for all MTP layers."""
        sharded_state_dict: ShardedStateDict = {}
        layer_prefix = f"{prefix}layers."
        for layer in self.layers:
            offset = get_mtp_layer_offset(self.config, self.vp_stage)
            sharded_prefix = f"{layer_prefix}{layer.layer_number - 1}."
            state_dict_prefix = f"{layer_prefix}{layer.layer_number - 1 - offset}."
            sharded_pp_offset: list = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(
                layer_sharded_state_dict, state_dict_prefix, sharded_prefix
            )
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict
