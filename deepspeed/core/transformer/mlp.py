# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""MLP — SwiGLU / GeGLU / GELU feed-forward network with TP sharding.

Ported from Megatron-LM megatron/core/transformer/mlp.py (17 commits,
M2312 → M4013).

Key evolution through the commit history:
  M2312 (2e29a5e1e) – quick_geglu activation for gpt-oss [4/5]
  M2346 (a329dd6da) – Enable bias in expert MLP [gpt-oss 5/5]
  M2814 (b51db3e07) – Support latent MoEs (moe_latent_size)
  M2837 (5ab481cb4) – Remove flattened_range code from distrib optimizer
  M2856 (5f5741db9) – Replace global parallel state w/ explicit pg params
  M2879 (f19b59eed) – NVLS fused RS+residual+RMSNorm+AG kernel
  M2886 (30694e0cd) – Refit prep 3 (sharded_state_dict_default)
  M2919 (1eed1d24f) – Typing pass
  M3078 (10c6f010e) – Remove padding token from MoE routing loss
  M3086 (4cfaa7d59) – Revert above
  M3127 (71c49b56d) – Fix for PR-2142 (padding token calc)
  M3138 (190f5b663) – Move kitchen extension to private repo
  M3253 (55198ba56) – Replace ModuleSpec with Protocols for MLP inputs
  M3890 (fa9c71454) – Handle SSM sharded tensor merge OOM with CPU fallback
  M3926 (5e3151416) – Protocol for MLP layer of TransformerLayer
  M4000 (32a7e46c7) – Use sharded_state_dict_default in MLP.sharded_state_dict
  M4013 (4c6360260) – FP4 param gather for NVFP4 recipe

Activation function evolution:
  GELU (original Megatron) → SwiGLU (LLaMA, Mistral) → GeGLU / quick-geglu
  (Qwen variants) → gated_linear_unit=True with configurable activation_func.

SwiGLU structure (gated_linear_unit=True):
  fc1 produces [gate_proj | up_proj] interleaved along output dim
  activation: SiLU(gate) * up  (stride=2 checkpoint-friendly)
  fc2 down-projects the result

GELU structure (gated_linear_unit=False):
  fc1 projects hidden → ffn_hidden_size
  activation: GELU(fc1_out)
  fc2 down-projects

DES-LOC integration
-------------------
MLP logs its assigned GPU tier (derived from layer_number) at construction.
The tier does not alter forward-pass logic; routing is handled externally.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig

# DES-LOC requirement: use deepspeed.comm instead of torch.distributed
try:
    import deepspeed.comm as dist
except ImportError:
    import torch.distributed as dist  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_tp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_group():
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except Exception:
        return None


def _get_tensor_model_parallel_group_if_none(tp_group, is_expert: bool = False):
    if tp_group is not None:
        return tp_group
    return _get_tp_group()


# ---------------------------------------------------------------------------
# Activation helpers (fused and non-fused)
# ---------------------------------------------------------------------------

def _bias_gelu_impl(x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """GELU with optional fused bias add."""
    if bias is not None:
        x = x + bias
    return F.gelu(x)


def _bias_swiglu_impl(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    fp8_input_store: bool = False,
    cpu_offloading: bool = False,
) -> torch.Tensor:
    """SiLU-gated linear unit with optional bias add.

    x is the concatenated [gate | up] tensor from fc1.
    This splits along the last dim and applies: SiLU(gate) * up.
    """
    if bias is not None:
        x = x + bias
    gate, up = torch.chunk(x, 2, dim=-1)
    return F.silu(gate) * up


def _bias_geglu_impl(
    x: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """GELU-gated linear unit with optional bias add."""
    if bias is not None:
        x = x + bias
    gate, up = torch.chunk(x, 2, dim=-1)
    return F.gelu(gate) * up


def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick-GELU: sigmoid approximation (x * sigmoid(1.702 * x))."""
    return x * torch.sigmoid(1.702 * x)


# Insight I9: FP32 aux_loss (Megatron M3394)
# Megatron M3394 found that computing aux_loss sigmoid scores in BF16 causes
# numerical instability at scale: small logit differences get flushed to zero in
# BF16 (11-bit mantissa) and the resulting uniform scores defeat load balancing.
# This is acutely worse in DES-LOC because A6000 (narrower memory bandwidth)
# accumulates BF16 rounding error faster than H100, so the two tiers would
# diverge in their routing distributions even for the same logits.
# Fix: always cast router_logits to FP32 before sigmoid.  The cost is a single
# upcast per router call — negligible compared to GEMM — but the stability gain
# is significant, especially in early training when logits are small and noisy.
def moe_router_sigmoid_fp32(router_logits: torch.Tensor) -> torch.Tensor:
    """Compute MoE routing sigmoid scores in FP32 for numerical stability.

    Always upcasts to FP32 regardless of input dtype (BF16 / FP16 / FP32).
    The output is kept in FP32 so that the subsequent aux_loss computation
    inherits the full-precision scores; callers that need BF16 output for
    dispatch weight application should cast back explicitly after loss accounting.

    Example::

        # In BF16 forward pass
        scores = moe_router_sigmoid_fp32(router_logits)  # FP32
        aux_loss = (scores * token_frac).sum()            # FP32 loss
        scores_bf16 = scores.to(router_logits.dtype)      # back to BF16 for dispatch

    Args:
        router_logits: Raw router logits of any dtype, shape [T, E].

    Returns:
        Sigmoid-normalised routing scores in FP32, shape [T, E].
    """
    # Insight I9: FP32 aux_loss (Megatron M3394)
    # Cast to FP32 before sigmoid to avoid BF16 precision loss in aux_loss.
    scores = torch.sigmoid(router_logits.float())  # FP32
    return scores



def _bias_quick_geglu_impl(
    x: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Quick-GELU gated linear unit (Qwen3 style)."""
    if bias is not None:
        x = x + bias
    gate, up = torch.chunk(x, 2, dim=-1)
    return _quick_gelu(gate) * up


# ---------------------------------------------------------------------------
# Protocol interfaces (M3253: replace ModuleSpec with Protocols)
# ---------------------------------------------------------------------------

class TEActivationFunctionInterface(Protocol):
    """Interface for activation_function module in MLP.

    Ported from Megatron-LM mlp.py — defines the contract for TE-based
    activation function modules (e.g. TE's SwiGLU / GeGLU fused kernels).
    """

    def forward(self, input_: torch.Tensor, /) -> torch.Tensor:
        """Forward method for activation_function module."""
        ...


class TEActivationFunctionBuilder(Protocol):
    """Protocol for activation_function module in MLP.

    Ported from Megatron-LM mlp.py — builder callable for TE activation
    function modules.
    """

    def __call__(self, *, config: TransformerConfig) -> TEActivationFunctionInterface:
        """Builds an activation function module for MLP."""
        ...


class LinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        # From Megatron M2967: in FP8 MoE grouped GEMM, bias MUST be applied
        # BEFORE unpadding. Post-unpad bias corrupts FP8 amax statistics
        # (padding activations pollute quantization scale computation).
        # Correct: gemm(padded) -> apply_bias -> unpad
        # Wrong:   gemm(padded) -> unpad -> apply_bias  <- BUG
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class LinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        stride: int = 1,
        name: str | None = None,
    ) -> LinearFc1Interface:
        ...


class LinearFc2Interface(Protocol):
    """Interface for linear_fc2 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class LinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        name: str | None = None,
    ) -> LinearFc2Interface:
        ...


@dataclass
class MLPSubmodules:
    """Dataclass for ModuleSpecs of MLP submodules.

    Contains linear_fc1, optional activation function, and linear_fc2.
    """

    linear_fc1: LinearFc1Builder
    linear_fc2: LinearFc2Builder
    activation_func: Optional[object] = None
    """Builder for an activation function module; only used when
    config.use_te_activation_func is True."""


# ---------------------------------------------------------------------------
# Simple self-contained linear layer for non-TE path
# ---------------------------------------------------------------------------

class _NativeLinear(nn.Module):
    """Native PyTorch linear layer compatible with LinearFc1/Fc2 interfaces.

    Used as the default when Transformer Engine is not available.

    This is the non-TE fallback that handles column-parallel (fc1) and
    row-parallel (fc2) patterns manually without Megatron's tensor_parallel
    module.  In DES-LOC heterogeneous clusters, A6000 tiers often cannot
    use TE (SM86, no FP8 hardware), so this layer handles their MLP compute
    at full precision.

    TP sharding convention (mirrors Megatron ColumnParallelLinear /
    RowParallelLinear):
      * ``tp_parallel_dim=0`` (column-parallel / fc1):
        Weight shape ``[out/tp, in]``.  Each TP rank owns a contiguous
        slice of output features.  No all-reduce needed after fc1;
        all-reduce happens after fc2.
      * ``tp_parallel_dim=1`` (row-parallel / fc2):
        Weight shape ``[out, in/tp]``.  Each TP rank owns a contiguous
        slice of input features.  All-reduce is applied after fc2 to sum
        partial outputs across TP ranks.
      * ``None``: no TP sharding (full weight on every rank).

    Stride support (M2312 / SwiGLU checkpoint portability):
      When ``stride > 1`` (typically 2 for SwiGLU), the output tensor of
      fc1 is ``[..., out_per_tp * stride]``.  The stride annotation on the
      weight tensor is needed so that Megatron's distributed optimizer can
      correctly re-shard the fc1 weights when changing TP size at checkpoint
      load time.

    Args:
        in_features: Input feature dimension (per-TP slice for row-parallel).
        out_features: Output feature dimension (per-TP slice for col-parallel).
        bias: Whether to include a bias term.
        tp_parallel_dim: 0 = column-parallel, 1 = row-parallel, None = no TP.
        stride: Stride factor for SwiGLU gate/up interleaving (default 1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_parallel_dim: Optional[int] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.tp_parallel_dim = tp_parallel_dim
        self.stride = stride
        if tp_parallel_dim is not None:
            self.linear.weight.tensor_model_parallel = True  # type: ignore[attr-defined]
            self.linear.weight.partition_dim = tp_parallel_dim  # type: ignore[attr-defined]
            if stride > 1:
                self.linear.weight.partition_stride = stride  # type: ignore[attr-defined]

    def forward(
        self, hidden_states: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Linear forward returning ``(output, bias)`` pair.

        Returns bias separately (not added to output) so that callers can
        fuse bias + activation in a single kernel when
        ``config.bias_activation_fusion`` is True.  This matches the
        ColumnParallelLinear / RowParallelLinear interface from Megatron.

        Args:
            hidden_states: Input tensor of any shape ending in ``in_features``.

        Returns:
            ``(output, bias)`` where bias is ``None`` if no bias is used.
        """
        out = F.linear(hidden_states, self.linear.weight)
        bias = self.linear.bias if (hasattr(self.linear, 'bias') and self.linear.bias is not None) else None
        return out, bias

    def backward_dw(self) -> None:
        """Weight gradient backward (no-op; handled by autograd)."""
        pass

    def extra_repr(self) -> str:
        return (
            f"in={self.linear.in_features}, out={self.linear.out_features}, "
            f"bias={self.linear.bias is not None}, "
            f"tp_dim={self.tp_parallel_dim}, stride={self.stride}"
        )


class _ColumnParallelLinear(_NativeLinear):
    """Column-parallel linear layer (fc1 / QKV projection style).

    Equivalent to Megatron's ``ColumnParallelLinear`` but without the TE
    dependency.  Partitions the output dimension across TP ranks.

    Each TP rank holds ``out_features // tp_size`` output rows, so the
    local weight shape is ``[out_features // tp_size, in_features]``.

    No all-reduce is needed after this layer (the partial outputs are
    consumed by the activation function and then ``_RowParallelLinear``
    which performs the all-reduce).

    For SwiGLU (``stride=2``), the output has shape
    ``[..., 2 * ffn_hidden // tp_size]`` interleaved as
    ``[gate_0, up_0, gate_1, up_1, ...]`` along the last dimension.

    Args:
        in_features: Full input size.
        out_features_per_tp: Output size *per TP rank* (= full_out // tp_size).
        bias: Whether to use bias.
        stride: 1 for GELU, 2 for SwiGLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_tp: int,
        bias: bool = False,
        stride: int = 1,
    ) -> None:
        super().__init__(
            in_features, out_features_per_tp, bias=bias,
            tp_parallel_dim=0, stride=stride,
        )


class _RowParallelLinear(_NativeLinear):
    """Row-parallel linear layer (fc2 / output projection style).

    Equivalent to Megatron's ``RowParallelLinear`` but without TE.
    Partitions the input dimension across TP ranks; each rank computes a
    partial dot-product over ``in_features // tp_size`` input columns, then
    an all-reduce sums the partial outputs.

    The all-reduce is performed by the parent MLP.forward() rather than
    inside this module, matching the pattern in Megatron where the caller
    controls when the all-reduce happens (for overlap with backward pass).

    Args:
        in_features_per_tp: Input size *per TP rank* (= full_in // tp_size).
        out_features: Full output size.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features_per_tp: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_features_per_tp, out_features, bias=bias, tp_parallel_dim=1
        )


def _build_native_fc1(
    input_size: int,
    output_size: int,
    *,
    bias: bool,
    is_expert: bool,
    stride: int = 1,
    **_kwargs,
) -> _ColumnParallelLinear:
    """Build a column-parallel fc1 linear layer (no TE fallback).

    The output_size is already the per-TP slice (caller has divided by tp_size).
    """
    return _ColumnParallelLinear(input_size, output_size, bias=bias, stride=stride)


def _build_native_fc2(
    input_size: int,
    output_size: int,
    *,
    bias: bool,
    **_kwargs,
) -> _RowParallelLinear:
    """Build a row-parallel fc2 linear layer (no TE fallback).

    The input_size is already the per-TP slice (caller has divided by tp_size).
    """
    return _RowParallelLinear(input_size, output_size, bias=bias)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(MegatronModule):
    """Feed-forward network with configurable activation and TP sharding.

    Two main variants:
      * SwiGLU / GeGLU (``gated_linear_unit=True``): fc1 output is double-wide
        and split into gate+up paths before the activation.
      * GELU (``gated_linear_unit=False``): standard two-layer FFN.

    Tensor-parallel sharding:
      * fc1: column-parallel (output dim sharded across TP ranks).
      * fc2: row-parallel (input dim sharded, output all-reduced).

    MoE latent support (M2814):
      When ``config.moe_latent_size`` is set and this is a routed expert,
      fc1 input → latent projection → ffn, and fc2 output → latent.

    From Megatron M2272 (30751977f): Fused MLP as subclass of unfused MLP —
      the fused TE-based MLP now inherits from this class so that checkpoint
      loading, sharded_state_dict, and tier-assignment logic are shared.
      In DES-LOC this matters because the H100 tier can use the TE fused path
      while A6000 (SM86, no TE FP8) falls back to the unfused path; both must
      produce identical state-dict keys for cross-tier checkpoint compatibility.

    DES-LOC integration:
      Logs tier assignment at construction via ``config.get_layer_tier()``.

    Args:
        config: TransformerConfig.
        submodules: Optional MLPSubmodules.  When None, uses native nn.Linear.
        is_expert: True if this MLP is a routed MoE expert.
        input_size: Override input hidden size (default: config.hidden_size).
        ffn_hidden_size: Override FFN hidden size (default: config.ffn_hidden_size).
        tp_group: Explicit TP process group.  None → auto from parallel_state.
        layer_number: 1-based global layer index for DES-LOC logging.
        name: Module instance name (passed top-down from parent).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MLPSubmodules] = None,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        layer_number: int = 0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.is_expert = is_expert

        self.input_size = input_size if input_size is not None else config.hidden_size

        self.tp_group = _get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        tp_size = _get_tp_world_size() if self.tp_group is None else (
            self.tp_group.size() if hasattr(self.tp_group, "size") else 1
        )

        # Resolve FFN hidden size
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError(
                    "MoE MLP requires `ffn_hidden_size`, but it was not provided."
                )
            if config.ffn_hidden_size is None:
                raise ValueError(
                    "MLP requires ffn_hidden_size; set it in TransformerConfig or pass directly."
                )
            warnings.warn(
                "MLP: using config.ffn_hidden_size as ffn_hidden_size.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = config.ffn_hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # MoE latent MLP (M2814)
        use_latent_size = (
            getattr(config, "moe_latent_size", None) is not None and is_expert
        )

        # SwiGLU: double the first projection width
        fc1_out_size = ffn_hidden_size
        if config.gated_linear_unit:
            fc1_out_size = ffn_hidden_size * 2
            fc1_stride = 2  # for correct weight resharding across TP sizes
            use_kitchen = getattr(config, "use_kitchen", False)
            if use_kitchen:
                fc1_stride = 1  # Kitchen Linear doesn't support stride != 1
        else:
            fc1_stride = 1

        # Per-TP-rank output size of fc1
        self._fc1_out_per_tp = fc1_out_size // tp_size
        self._fc2_in_per_tp = ffn_hidden_size // tp_size

        # Build fc1
        if submodules is not None and submodules.linear_fc1 is not None:
            fc1_input = (
                self.input_size
                if not use_latent_size
                else config.moe_latent_size
            )
            self.linear_fc1 = submodules.linear_fc1(
                fc1_input,
                fc1_out_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=config.add_bias_linear,
                skip_bias_add=True,
                is_expert=is_expert,
                tp_comm_buffer_name="fc1",
                tp_group=tp_group,
                stride=fc1_stride,
                name=(name + ".linear_fc1") if name else None,
            )
        else:
            # Native fallback: column-parallel
            fc1_input_size = self.input_size
            self.linear_fc1 = _NativeLinear(
                fc1_input_size, self._fc1_out_per_tp,
                bias=config.add_bias_linear, tp_parallel_dim=0,
            )

        # Activation function
        use_te_activation = getattr(config, "use_te_activation_func", False)
        if (
            use_te_activation
            and submodules is not None
            and submodules.activation_func is not None
        ):
            self.activation_func = submodules.activation_func(config=config)
        else:
            self.activation_func = config.activation_func

        # Build fc2
        if submodules is not None and submodules.linear_fc2 is not None:
            fc2_out_size = (
                config.hidden_size
                if not use_latent_size
                else config.moe_latent_size
            )
            self.linear_fc2 = submodules.linear_fc2(
                ffn_hidden_size,
                fc2_out_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=is_expert,
                tp_comm_buffer_name="fc2",
                tp_group=tp_group,
                name=(name + ".linear_fc2") if name else None,
            )
        else:
            # Native fallback: row-parallel
            self.linear_fc2 = _NativeLinear(
                self._fc2_in_per_tp, config.hidden_size,
                bias=config.add_bias_linear, tp_parallel_dim=1,
            )

        # DES-LOC: log tier assignment
        if layer_number > 0:
            tier = config.get_layer_tier(layer_number - 1)
            if tier is not None:
                logger.debug(
                    "MLP layer %d → DES-LOC tier: %s", layer_number, tier.upper()
                )

    # ------------------------------------------------------------------
    # Activation dispatch
    # ------------------------------------------------------------------

    def _apply_activation(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        per_token_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply activation to fc1 output.

        Handles gated_linear_unit (SwiGLU / GeGLU / quick-GeGLU) and
        standard GELU paths, with optional fused-bias kernels.

        Args:
            x: fc1 output ``[s, b, ffn_per_tp]`` (or ``[..., 2*ffn_per_tp]``
               when gated_linear_unit=True).
            bias: Optional bias from fc1 (may be None).
            per_token_scale: Optional per-token scale for MoE token-weighted MLP.

        Returns:
            Activated intermediate ``[s, b, ffn_per_tp]``.
        """
        use_te_activation = getattr(self.config, "use_te_activation_func", False)
        bias_act_fusion = getattr(self.config, "bias_activation_fusion", False)

        if use_te_activation:
            if bias is not None:
                x = x + bias
            x = self.activation_func(x)
            if per_token_scale is not None:
                orig_dtype = x.dtype
                x = x * per_token_scale.unsqueeze(-1)
                x = x.to(orig_dtype)
            return x

        if bias_act_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    x = _bias_swiglu_impl(x, bias)
                    orig_dtype = x.dtype
                    x = x * per_token_scale.unsqueeze(-1)
                    x = x.to(orig_dtype)
                    return x
                elif (
                    self.activation_func == _quick_gelu
                    and self.config.gated_linear_unit
                ):
                    x = _bias_quick_geglu_impl(x, bias)
                    orig_dtype = x.dtype
                    x = x * per_token_scale.unsqueeze(-1)
                    x = x.to(orig_dtype)
                    return x
                else:
                    raise ValueError(
                        "Only swiglu and quick_gelu support per_token_scale fusion in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        return _bias_geglu_impl(x, bias)
                    else:
                        if bias is not None:
                            x = x + bias
                        return F.gelu(x)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    return _bias_swiglu_impl(x, bias)
                else:
                    raise ValueError("Only gelu and swiglu support bias_activation_fusion.")

        # Non-fused path
        if bias is not None:
            x = x + bias

        if self.config.gated_linear_unit:
            clamp_val = getattr(self.config, "activation_func_clamp_value", None)
            glu_linear_offset = getattr(self.config, "glu_linear_offset", 0)

            gate, up = torch.chunk(x, 2, dim=-1)
            if clamp_val is not None:
                gate = gate.clamp(min=None, max=clamp_val)
                up = up.clamp(min=-clamp_val, max=clamp_val)
            x = self.activation_func(gate) * (up + glu_linear_offset)
        else:
            x = self.activation_func(x)

        if per_token_scale is not None:
            orig_dtype = x.dtype
            x = x * per_token_scale.unsqueeze(-1)
            x = x.to(orig_dtype)

        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_token_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the MLP.

        Args:
            hidden_states: ``[s, b, h]``
            per_token_scale: Optional per-token scale ``[s*b, 1]`` for MoE.

        Returns:
            (output, output_bias): output is ``[s, b, h]``, output_bias is None
            unless add_bias_linear=True (in which case it is additive).
        """
        # --- fc1 [s, b, h] → [s, b, ffn_per_tp] (or 2x for SwiGLU) ------
        if isinstance(self.linear_fc1, _NativeLinear):
            intermediate_parallel, bias_parallel = self.linear_fc1.forward(hidden_states)
        else:
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        # --- activation ---------------------------------------------------
        intermediate_parallel = self._apply_activation(
            intermediate_parallel, bias_parallel, per_token_scale
        )

        # --- fc2 [s, b, ffn_per_tp] → [s, b, h] --------------------------
        if isinstance(self.linear_fc2, _NativeLinear):
            output, output_bias = self.linear_fc2.forward(intermediate_parallel)
        else:
            output, output_bias = self.linear_fc2(cast(torch.Tensor, intermediate_parallel))

        # Row-parallel all-reduce
        tp_group = self.tp_group
        if tp_group is None:
            tp_group = _get_tp_group()
        tp_size = _get_tp_world_size()
        if tp_size > 1 and tp_group is not None:
            dist.all_reduce(output, group=tp_group)

        # MoE: if bias present and expert, add to output directly
        if per_token_scale is not None and output_bias is not None:
            output = output + output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Trigger weight-gradient updates for both fc layers."""
        if hasattr(self.linear_fc2, "backward_dw"):
            self.linear_fc2.backward_dw()
        if hasattr(self.linear_fc1, "backward_dw"):
            self.linear_fc1.backward_dw()

    def set_layer_number(self, layer_number: int) -> None:
        """Set the 1-based global layer number for this MLP.

        Called by ``TransformerLayer.__init__`` after MLP construction so that
        the DES-LOC tier logging and fine-grained offloading hooks have access
        to the correct layer index.  Without this, MLP's layer_number would
        always be 0 (the default from the fallback MLP constructor path that
        doesn't receive layer_number from the spec pattern).

        Args:
            layer_number: 1-based global layer index.
        """
        if self.layer_number == 0 and layer_number > 0:
            self.layer_number = layer_number
            # Update DES-LOC tier logging now that we have the real layer number.
            tier = self.config.get_layer_tier(layer_number - 1)
            if tier is not None:
                logger.debug(
                    "MLP layer %d → DES-LOC tier: %s (updated via set_layer_number)",
                    layer_number, tier.upper(),
                )

    def count_parameters(self, *, exclude_shared: bool = False) -> int:
        """Count the total number of parameters in this MLP.

        Useful for the DES-LOC engine to balance parameter counts across
        tier boundaries and for logging.

        Args:
            exclude_shared: If True, count only non-shared (local) parameters.
                Shared expert parameters in MoE models should be excluded when
                counting routed-expert parameter budgets.

        Returns:
            Total parameter count (all elements, not just the TP-local slice).
        """
        total = 0
        for name, param in self.named_parameters():
            if exclude_shared and getattr(param, 'shared', False):
                continue
            # Scale up by TP size for column-parallel params (dim 0 sharded)
            tp_dim = getattr(param, 'partition_dim', None)
            tp_size = _get_tp_world_size()
            count = param.numel()
            if tp_dim is not None and tp_size > 1:
                count *= tp_size  # approximate full parameter count
            total += count
        return total

    def get_parameter_groups(self) -> dict:
        """Return parameter groups for DES-LOC tiered optimizer.

        Groups parameters by their role (fc1/gate, fc1/up, fc2) to allow
        the DES-LOC optimizer to apply different weight decay, learning rate
        scaling, or sync periods per group.

        Returns:
            Dict mapping group name → list of parameters::

                {
                    "fc1": [fc1_weight, fc1_bias],        # gate+up (SwiGLU) or full
                    "fc2": [fc2_weight, fc2_bias],        # down projection
                    "activation": [],                      # learnable activation params
                }
        """
        groups: dict = {"fc1": [], "fc2": [], "activation": []}
        for name, param in self.linear_fc1.named_parameters():
            groups["fc1"].append(param)
        for name, param in self.linear_fc2.named_parameters():
            groups["fc2"].append(param)
        if isinstance(self.activation_func, torch.nn.Module):
            for name, param in self.activation_func.named_parameters():
                groups["activation"].append(param)
        return groups

    # ------------------------------------------------------------------
    # Sharded state dict (M4000)
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Return sharded state dict.

        For SwiGLU (gated_linear_unit=True), fc1 weights are split into
        gate and up halves so that checkpoints are portable across TP sizes.
        """
        sharded_state_dict: dict = {}
        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)

        for name, module in self._modules.items():
            sub_sd = self._module_sharded_state_dict(
                module, f"{prefix}{name}.", sharded_offsets, metadata
            )
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = _apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)

        return sharded_state_dict

    def _module_sharded_state_dict(
        self, module, prefix, sharded_offsets, metadata
    ) -> dict:
        """Helper to collect state dict from a sub-module."""
        if hasattr(module, "sharded_state_dict"):
            return module.sharded_state_dict(prefix, sharded_offsets, metadata)
        # Fallback: plain state dict
        return {f"{prefix}{k}": v for k, v in module.state_dict(prefix="").items()}

    # ------------------------------------------------------------------
    # Class method helper (M3926)
    # ------------------------------------------------------------------

    @classmethod
    def as_mlp_submodule(
        cls,
        submodules: MLPSubmodules,
        config: TransformerConfig,
        pg_collection: object,
        is_mtp_layer: bool,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "MLP":
        """Build an MLP as a TransformerLayer's mlp submodule.

        This is the Protocol-based factory used since M3926.
        """
        tp_group = pg_collection.tp if hasattr(pg_collection, "tp") else None
        return cls(
            config=config,
            submodules=submodules,
            tp_group=tp_group,
            is_expert=is_expert,
            input_size=input_size,
            ffn_hidden_size=ffn_hidden_size,
            name=name,
        )


# ---------------------------------------------------------------------------
# SwiGLU sharded factory (for checkpoint portability across TP sizes)
# ---------------------------------------------------------------------------

def _apply_swiglu_sharded_factory(
    original_tensor,
    sharded_offsets: tuple,
    singleton_local_shards: bool = False,
) -> object:
    """Wrap a SwiGLU fc1 tensor so it can be saved/loaded across TP sizes.

    The fc1 weight has shape ``[2 * ffn/tp, hidden]``.  We split it into
    gate (first half) and up (second half) for checkpoint compatibility.

    When the loaded checkpoint has a different TP degree, the gate and up
    halves can be sharded independently and then re-interleaved.

    This implementation mirrors Megatron's ``apply_swiglu_sharded_factory``
    (M4000 era).  When Megatron's ``ShardedTensorFactory`` is available (i.e.
    running within the Megatron dist-checkpoint framework), it constructs the
    full factory object so that the checkpoint can be saved/loaded across
    different TP sizes.  When running standalone (plain PyTorch tensors), it
    returns the tensor unchanged because cross-TP resharding is not needed.

    Args:
        original_tensor: The state dict tensor (plain or ShardedTensor).
        sharded_offsets: PP/TP offset tuple from the caller.
        singleton_local_shards: Whether to use singleton shard format.

    Returns:
        A ShardedTensorFactory (Megatron path) or the original tensor (fallback).
    """
    # Try to use Megatron's dist-checkpoint infrastructure for full cross-TP
    # checkpoint portability.
    try:
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.mapping import (
            ReplicaId,
            ShardedTensorFactory,
        )
        from megatron.core.transformer.utils import cat_with_oom_fallback

        swiglu_shard_axis = 0
        prepend_axis_num = len(sharded_offsets)
        original_shape = original_tensor.local_shape
        local_axis_size = original_shape[swiglu_shard_axis]
        assert (
            original_tensor.global_offset[swiglu_shard_axis + prepend_axis_num]
            % local_axis_size == 0
        )
        rank_offset = (
            original_tensor.global_offset[swiglu_shard_axis + prepend_axis_num]
            // local_axis_size
        )
        axis_frag = original_tensor.axis_fragmentations[
            swiglu_shard_axis + prepend_axis_num
        ]

        @torch.no_grad()
        def sh_ten_build_fn(key, t, replica_id, flattened_range):
            if singleton_local_shards:
                offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
                offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
                w_key = f'{key}_w'
                v_key = f'{key}_v'
            else:
                offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
                offset_v = (
                    swiglu_shard_axis + prepend_axis_num,
                    rank_offset + axis_frag,
                    axis_frag * 2,
                )
                w_key = key
                v_key = key

            tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    w_key, tensor_w, *sharded_offsets, offset_w,
                    replica_id=replica_id, prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    v_key, tensor_v, *sharded_offsets, offset_v,
                    replica_id=replica_id, prepend_axis_num=prepend_axis_num,
                ),
            ]

        return ShardedTensorFactory(
            original_tensor.key,
            original_tensor.data,
            sh_ten_build_fn,
            cat_with_oom_fallback,
            original_tensor.replica_id,
            flattened_range=original_tensor.flattened_range,
        )
    except (ImportError, AttributeError):
        # Running without Megatron's dist-checkpoint framework — return
        # the tensor unchanged.  Cross-TP checkpoint resharding is not
        # supported in this mode; the user must use matching TP sizes.
        return original_tensor


# Megatron-compatible public name (M4000)
apply_swiglu_sharded_factory = _apply_swiglu_sharded_factory
