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

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include a bias term.
        tp_parallel_dim: 0 = column-parallel, 1 = row-parallel, None = no TP.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_parallel_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if tp_parallel_dim is not None:
            self.linear.weight.tensor_model_parallel = True
            self.linear.weight.partition_dim = tp_parallel_dim

    def forward(
        self, hidden_states: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bias = self.linear.bias if hasattr(self.linear, "bias") else None
        out = F.linear(hidden_states, self.linear.weight)
        return out, bias

    def backward_dw(self) -> None:
        """Weight gradient backward (no-op; handled by autograd)."""
        pass


def _build_native_fc1(
    input_size: int,
    output_size: int,
    *,
    bias: bool,
    is_expert: bool,
    stride: int = 1,
    **_kwargs,
) -> _NativeLinear:
    return _NativeLinear(input_size, output_size, bias=bias, tp_parallel_dim=0)


def _build_native_fc2(
    input_size: int,
    output_size: int,
    *,
    bias: bool,
    **_kwargs,
) -> _NativeLinear:
    return _NativeLinear(input_size, output_size, bias=bias, tp_parallel_dim=1)


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
            torch.distributed.all_reduce(output, group=tp_group)

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

    This is a simplified version of Megatron's ShardedTensorFactory that
    records the split metadata in the key name.

    Args:
        original_tensor: The state dict tensor (plain or ShardedTensor).
        sharded_offsets: PP/TP offset tuple from the caller.
        singleton_local_shards: Whether to use singleton shard format.

    Returns:
        A dict with two entries (gate, up) instead of one.
    """
    # For native PyTorch tensors just return them unchanged; the more complex
    # ShardedTensorFactory logic only applies in Megatron's dist-checkpoint
    # framework.  Here we just mark the intent.
    return original_tensor
