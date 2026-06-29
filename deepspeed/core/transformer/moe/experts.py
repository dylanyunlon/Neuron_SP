# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GroupedMLP + SequentialMLP expert layers.

Ported from Megatron-LM megatron/core/transformer/moe/experts.py.
All ``megatron.core`` imports have been replaced with ``deepspeed.core``
equivalents.  Megatron-specific features that have no deepspeed.core
counterpart (Transformer Engine grouped-linear, FlashInfer, MXFP8, paged
stash, fine-grained activation offloading) are guarded behind try/import
blocks or config flags so the module loads cleanly without those optional
dependencies.

Classes exported
----------------
GroupedMLPSubmodules   – dataclass for TEGroupedMLP submodule specs
TEGroupedMLP           – grouped (batched) expert forward; falls back to a
                         pure-PyTorch path when TE is not available
SequentialMLP          – sequential per-expert forward; reference implementation

Design decisions
----------------
* ProcessGroupCollection is imported from deepspeed.core.transformer.moe.token_dispatcher
  (the minimal stub that matches Megatron's interface).
* tensor_parallel.CheckpointWithoutOutput comes from deepspeed.core.tensor_parallel.
* MegatronModule / TransformerConfig come from deepspeed.core.transformer.
* Sharding helpers (sharded_state_dict_default, ensure_metadata_has_dp_cp_group,
  replace_prefix_for_sharding, apply_swiglu_sharded_factory) are implemented
  locally as thin wrappers because deepspeed.core has not yet ported the full
  dist_checkpointing stack.
* TE-specific paths (TEGroupedLinear, FP8 padding, fused-ops, inference grouped
  backends) are preserved structurally but guarded by ``HAVE_TE`` / try blocks
  so non-TE environments degrade gracefully.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from math import ceil
from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# deepspeed.core equivalents of megatron.core imports
# ---------------------------------------------------------------------------
from deepspeed.core import tensor_parallel
from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.mlp import MLP, MLPSubmodules
from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.transformer.moe.token_dispatcher import ProcessGroupCollection

# ---------------------------------------------------------------------------
# Optional: Transformer Engine
# ---------------------------------------------------------------------------
try:
    import transformer_engine as te  # type: ignore[import]
    from transformer_engine.pytorch import GroupedLinear as TEGroupedLinear  # type: ignore
    HAVE_TE = True
except ImportError:
    te = None  # type: ignore[assignment]
    TEGroupedLinear = None  # type: ignore[assignment, misc]
    HAVE_TE = False

# TE FP8 padding helpers – only available with sufficiently new TE
try:
    from transformer_engine.pytorch.fp8 import Fp8Padding, Fp8Unpadding  # type: ignore
    _HAVE_TE_FP8_PAD = True
except ImportError:
    try:
        from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding  # type: ignore
        _HAVE_TE_FP8_PAD = True
    except ImportError:
        Fp8Padding = None  # type: ignore[assignment, misc]
        Fp8Unpadding = None  # type: ignore[assignment, misc]
        _HAVE_TE_FP8_PAD = False

# Optional: paged stash (MoE activation paging)
try:
    from deepspeed.core.transformer.moe.paged_stash import (  # type: ignore[import]
        get_paged_stash_context,
        paged_stash_group_commit,
        paged_stash_group_start,
    )
    _HAVE_PAGED_STASH = True
except ImportError:
    _HAVE_PAGED_STASH = False

    def get_paged_stash_context(*args, **kwargs):  # type: ignore[misc]
        return nullcontext()

    def paged_stash_group_start(x):  # type: ignore[misc]
        return x

    def paged_stash_group_commit(x, **kwargs):  # type: ignore[misc]
        return x


# Optional: fine-grained activation offloading
try:
    from deepspeed.core.pipeline_parallel.fine_grained_activation_offload import (  # type: ignore[import]
        FineGrainedActivationOffloadingInterface as off_interface,
    )
    _HAVE_FGAO = True
except ImportError:
    _HAVE_FGAO = False

    class _NoopOffInterface:
        """Transparent context manager / no-op for fine-grained offloading."""

        def __init__(self, enabled, tensor, name):
            self._tensor = tensor

        def __enter__(self):
            return self._tensor

        def __exit__(self, *_):
            return False

        @staticmethod
        def group_commit(tensor, name=None, forced_released_tensors=None):
            return tensor

    off_interface = _NoopOffInterface  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def _squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).pow(2)


def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


# Try to import from deepspeed.core first, fall back to local definitions
try:
    from deepspeed.core.fusions.fused_bias_geglu import fused_bias_swiglu  # type: ignore
    _weighted_bias_swiglu_impl = None
except ImportError:
    _weighted_bias_swiglu_impl = None

# weighted activation impls — pure-PyTorch fallbacks (no fused kernels)
def _weighted_bias_swiglu(x, bias, probs, *args):
    if bias is not None:
        x = x + bias
    mid = x.shape[-1] // 2
    gate, up = x[..., :mid], x[..., mid:]
    out = F.silu(gate) * up
    return out * probs


def _weighted_bias_quick_geglu(x, bias, probs, *args, **kwargs):
    if bias is not None:
        x = x + bias
    mid = x.shape[-1] // 2
    gate, up = x[..., :mid], x[..., mid:]
    out = _quick_gelu(gate) * up
    return out * probs


def _weighted_squared_relu_impl(x, probs):
    return _squared_relu(x) * probs


# Try to import optimized fused versions if available
try:
    from deepspeed.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl  # type: ignore
except ImportError:
    weighted_bias_swiglu_impl = _weighted_bias_swiglu  # type: ignore[assignment]

try:
    from deepspeed.core.fusions.fused_bias_geglu import weighted_bias_quick_geglu_impl  # type: ignore
except ImportError:
    weighted_bias_quick_geglu_impl = _weighted_bias_quick_geglu  # type: ignore[assignment]

try:
    from deepspeed.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl  # type: ignore
except ImportError:
    weighted_squared_relu_impl = _weighted_squared_relu_impl  # type: ignore[assignment]

# swiglu sharded factory for checkpointing
try:
    from deepspeed.core.transformer.mlp import _apply_swiglu_sharded_factory as apply_swiglu_sharded_factory  # type: ignore
except ImportError:
    def apply_swiglu_sharded_factory(sharded_tensor, sharded_offsets, singleton=False):  # type: ignore[misc]
        return sharded_tensor


# ---------------------------------------------------------------------------
# Quantization alignment helper (mirrors Megatron's get_align_size_for_quantization)
# ---------------------------------------------------------------------------
def _get_align_size(config: TransformerConfig) -> int:
    if getattr(config, "fp4", None):
        return 32
    if getattr(config, "fp8", None):
        return 16
    return 1


def _skip_routed_expert_padding(config: TransformerConfig) -> bool:
    """Return True when padding should be skipped (e.g. no FP8/FP4)."""
    return not (getattr(config, "fp8", None) or getattr(config, "fp4", None))


# ---------------------------------------------------------------------------
# Sharding helpers — thin local implementations
# (deepspeed.core has not ported the full dist_checkpointing stack)
# ---------------------------------------------------------------------------

def _ensure_metadata_has_dp_cp_group(metadata: Optional[dict]) -> dict:
    return metadata or {}


def _sharded_state_dict_default(module, prefix, sharded_offsets, metadata, tp_group=None):
    """Minimal sharding wrapper: delegates to module.sharded_state_dict if present."""
    if hasattr(module, "sharded_state_dict"):
        return module.sharded_state_dict(prefix, sharded_offsets, metadata)
    # Plain state dict with prefix
    return {f"{prefix}{k}": v for k, v in module.state_dict().items()}


def _replace_prefix_for_sharding(sub_sd: dict, old_prefix: str, new_prefix: str) -> None:
    """In-place rename keys in *sub_sd* from old_prefix to new_prefix."""
    keys_to_rename = [k for k in sub_sd if k.startswith(old_prefix)]
    for k in keys_to_rename:
        new_k = new_prefix + k[len(old_prefix):]
        sub_sd[new_k] = sub_sd.pop(k)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol / dataclass types
# ---------------------------------------------------------------------------

class GroupedLinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in TEGroupedMLP."""

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class GroupedLinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
        name: str | None = None,
    ) -> GroupedLinearFc1Interface:
        """Builds a linear_fc1 layer for TEGroupedMLP."""
        ...


class GroupedLinearFc2Interface(Protocol):
    """Protocol for linear_fc2 module in TEGroupedMLP."""

    def forward(
        self, intermediate_parallel: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class GroupedLinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
        name: str | None = None,
    ) -> GroupedLinearFc2Interface:
        """Builds a linear_fc2 layer for TEGroupedMLP."""
        ...


# TEActivationFunctionBuilder — optional (TE-only)
try:
    from deepspeed.core.transformer.mlp import TEActivationFunctionBuilder  # type: ignore
except ImportError:
    TEActivationFunctionBuilder = None  # type: ignore[assignment, misc]


@dataclass
class GroupedMLPSubmodules:
    """Dataclass for ModuleSpecs of TEGroupedMLP submodules.

    Holds builders for linear_fc1, an optional activation function, and
    linear_fc2.
    """

    linear_fc1: GroupedLinearFc1Builder
    linear_fc2: GroupedLinearFc2Builder
    activation_func: Optional[object] = None
    """Builder for a TE activation function module; only used when
    config.use_te_activation_func is True."""


# ---------------------------------------------------------------------------
# Helper: safe module apply (handles None modules)
# ---------------------------------------------------------------------------
def _apply_module(module, *args, **kwargs):
    """Call module(*args, **kwargs), handling None gracefully."""
    if module is None:
        raise ValueError("Attempted to call a None module.")
    return module(*args, **kwargs)


def _not_none(x, msg="Unexpected None"):
    if x is None:
        raise ValueError(msg)
    return x


# ---------------------------------------------------------------------------
# TEGroupedMLP
# ---------------------------------------------------------------------------

class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximise computational efficiency.

    When Transformer Engine is not available, falls back to a pure-PyTorch
    sequential path that preserves interface compatibility.

    Args:
        num_local_experts: Number of experts local to this rank.
        config: TransformerConfig instance.
        submodules: GroupedMLPSubmodules with fc1/fc2 builders.
        pg_collection: Process-group bundle (ep, expt_tp groups).
        name: Optional module name for checkpoint-friendly parameter naming.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size

        pg_collection = pg_collection or ProcessGroupCollection()
        assert not (
            self.config.add_bias_linear and getattr(config, "bias_dropout_fusion", False)
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp

        # Double output width for gated linear unit (GLU / SwiGLU)
        ffn_hidden_size = _not_none(
            self.config.moe_ffn_hidden_size,
            "config.moe_ffn_hidden_size must be set for TEGroupedMLP",
        )
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = submodules.linear_fc1(
            self.num_local_experts,
            self.input_size if self.config.moe_latent_size is None else self.config.moe_latent_size,
            ffn_hidden_size,
            config=self.config,
            init_method=_not_none(getattr(self.config, "init_method", None)),
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name="fc1",
            pg_collection=pg_collection,
            name=(name + ".linear_fc1") if name is not None else None,
        )

        # Activation function
        if (
            getattr(self.config, "use_te_activation_func", False)
            and submodules.activation_func is not None
        ):
            self.activation_func = submodules.activation_func(config=self.config)
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = submodules.linear_fc2(
            self.num_local_experts,
            _not_none(self.config.moe_ffn_hidden_size),
            (
                self.config.hidden_size
                if self.config.moe_latent_size is None
                else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=_not_none(getattr(self.config, "output_layer_init_method", None)),
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name="fc2",
            pg_collection=pg_collection,
            name=(name + ".linear_fc2") if name is not None else None,
        )

        # Fine-grained activation offloading flags
        _offload_modules = getattr(self.config, "offload_modules", None) or []
        self.offload_expert_fc1 = (
            getattr(self.config, "fine_grained_activation_offloading", False)
            and "expert_fc1" in _offload_modules
        )
        self.offload_moe_act = (
            getattr(self.config, "fine_grained_activation_offloading", False)
            and "moe_act" in _offload_modules
        )

        # Selective activation recompute
        _recompute_modules = getattr(self.config, "recompute_modules", None) or []
        self.activation_recompute = (
            getattr(self.config, "recompute_granularity", None) == "selective"
            and "moe_act" in _recompute_modules
        )

        # FP8 / FP4 quantization padding (requires TE)
        _use_fp8 = getattr(self.config, "fp8", None)
        _use_fp4 = getattr(self.config, "fp4", None)
        if _use_fp8 or _use_fp4:
            assert HAVE_TE and _HAVE_TE_FP8_PAD, "FP8/FP4 quantization requires Transformer Engine."
            self.quantization_padding = Fp8Padding(self.num_local_experts)
            self.quantization_unpadding = Fp8Unpadding(self.num_local_experts)

        # TE op-fuser path (lazy-initialised on first forward)
        self._with_fused_impl: bool = getattr(self.config, "use_transformer_engine_op_fuser", False)
        self._fused_ops: Optional[Tuple[torch.nn.Module]] = None
        if self._with_fused_impl:
            assert self._is_fused_impl_supported(), (
                "use_transformer_engine_op_fuser=True but fused GroupedMLP is not "
                "supported for this configuration."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def _is_fused_impl_supported(self) -> bool:
        """Check whether the TE op-fuser can handle this configuration."""
        if not HAVE_TE:
            return False
        try:
            from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
        except ImportError:
            return False

        if self.tp_group is not None and self.tp_group.size() > 1:
            return False
        if self.offload_expert_fc1 or self.offload_moe_act:
            return False
        if getattr(self.config, "moe_apply_probs_on_input", False):
            return False
        if not isinstance(self.linear_fc1, TEGroupedLinear):
            return False
        if not isinstance(self.linear_fc2, TEGroupedLinear):
            return False

        use_glu_fusion = self.config.gated_linear_unit and self.config.activation_func in (
            F.silu,
            _quick_gelu,
        )
        use_srelu_fusion = (
            self.config.activation_func == _squared_relu
            and getattr(self.config, "use_fused_weighted_squared_relu", False)
            and not self.config.gated_linear_unit
        )
        return use_glu_fusion or use_srelu_fusion

    @staticmethod
    def _remove_glu_interleaving(x: torch.Tensor, interleave_size: int) -> torch.Tensor:
        """Reorder interleaved GLU blocks so gate and linear halves are contiguous."""
        shape = x.size()
        x = x.reshape(-1, shape[-1] // (2 * interleave_size), 2, interleave_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(shape)
        return x

    # ------------------------------------------------------------------
    # Fused TE impl (preserved for completeness; guarded by HAVE_TE)
    # ------------------------------------------------------------------

    def _make_fused_ops(self) -> torch.nn.Module:  # pragma: no cover
        """Construct fused FC1 + activation + FC2 TE op sequence."""
        assert HAVE_TE, "_make_fused_ops requires Transformer Engine."
        ops = te.pytorch.ops.Sequential()

        fc1_single_grouped_weight = self.linear_fc1.single_grouped_weight
        fc2_single_grouped_weight = self.linear_fc2.single_grouped_weight
        fc1_single_grouped_bias = getattr(self.linear_fc1, "single_grouped_bias", False)
        fc2_single_grouped_bias = getattr(self.linear_fc2, "single_grouped_bias", False)
        fc1_weight_dtype = (
            self.linear_fc1.weight.dtype
            if fc1_single_grouped_weight
            else self.linear_fc1.weight0.dtype
        )
        fc2_weight_dtype = (
            self.linear_fc2.weight.dtype
            if fc2_single_grouped_weight
            else self.linear_fc2.weight0.dtype
        )
        fc1_delay_wgrad = getattr(self.linear_fc1, "delay_wgrad_compute", False)
        fc2_delay_wgrad = getattr(self.linear_fc2, "delay_wgrad_compute", False)

        op = te.pytorch.ops.GroupedLinear(
            self.linear_fc1.num_gemms,
            self.linear_fc1.in_features,
            self.linear_fc1.out_features,
            bias=self.linear_fc1.use_bias,
            device="meta",
            dtype=fc1_weight_dtype,
            accumulate_into_main_grad=getattr(self.linear_fc1, "fuse_wgrad_accumulation", False),
            single_grouped_weight=fc1_single_grouped_weight,
            single_grouped_bias=fc1_single_grouped_bias,
            delay_wgrad_compute=fc1_delay_wgrad,
        )
        if fc1_single_grouped_weight:
            setattr(op, "weight", getattr(self.linear_fc1, "weight"))
        for idx in range(self.linear_fc1.num_gemms):
            if not fc1_single_grouped_weight:
                setattr(op, f"weight{idx}", getattr(self.linear_fc1, f"weight{idx}"))
            if self.linear_fc1.use_bias and not fc1_single_grouped_bias:
                setattr(op, f"bias{idx}", getattr(self.linear_fc1, f"bias{idx}"))
        if self.linear_fc1.use_bias and fc1_single_grouped_bias:
            setattr(op, "bias", getattr(self.linear_fc1, "bias"))
        ops.append(op)

        # Activation op
        glu_interleave = getattr(self.config, "moe_mlp_glu_interleave_size", None)
        activation_recompute_in_mlp = bool(getattr(self, "activation_recompute", False))
        if self.config.activation_func == F.silu and self.config.gated_linear_unit:
            op = te.pytorch.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave)
        elif self.config.activation_func == _quick_gelu and self.config.gated_linear_unit:
            clamp = getattr(self.config, "activation_func_clamp_value", None)
            try:
                from transformer_engine.pytorch.ops import ScaledClampedQGeGLU
                op = ScaledClampedQGeGLU(glu_interleave_size=glu_interleave, limit=clamp)
            except ImportError:
                raise RuntimeError("ScaledClampedQGeGLU not found in Transformer Engine.")
        else:
            raise RuntimeError("_make_fused_ops: unsupported activation for fused path.")
        ops.append(op)

        op = te.pytorch.ops.GroupedLinear(
            self.linear_fc2.num_gemms,
            self.linear_fc2.in_features,
            self.linear_fc2.out_features,
            bias=self.linear_fc2.use_bias,
            device="meta",
            dtype=fc2_weight_dtype,
            accumulate_into_main_grad=getattr(self.linear_fc2, "fuse_wgrad_accumulation", False),
            single_grouped_weight=fc2_single_grouped_weight,
            single_grouped_bias=fc2_single_grouped_bias,
            delay_wgrad_compute=fc2_delay_wgrad,
        )
        if fc2_single_grouped_weight:
            setattr(op, "weight", getattr(self.linear_fc2, "weight"))
        for idx in range(self.linear_fc2.num_gemms):
            if not fc2_single_grouped_weight:
                setattr(op, f"weight{idx}", getattr(self.linear_fc2, f"weight{idx}"))
            if self.linear_fc2.use_bias and not fc2_single_grouped_bias:
                setattr(op, f"bias{idx}", getattr(self.linear_fc2, f"bias{idx}"))
        if self.linear_fc2.use_bias and fc2_single_grouped_bias:
            setattr(op, "bias", getattr(self.linear_fc2, "bias"))
        ops.append(op)

        ops.register_forward_pre_hook(self._make_fused_impl_pre_forward_hook())
        return ops

    def _make_fused_impl_pre_forward_hook(self) -> Callable:
        """Pre-forward hook that fires DDP all-gather hooks on sub-modules."""

        def forward_pre_hook(module, *_) -> None:
            for submodule in chain(self.linear_fc1.modules(), self.linear_fc2.modules()):
                for hook in submodule._forward_pre_hooks.values():
                    ret = hook(submodule, None)
                    if ret is not None:
                        raise RuntimeError(
                            f"Applying a fused implementation for {self.__class__.__name__}, "
                            f"but a {submodule.__class__.__name__} submodule "
                            "has a pre-forward hook that modifies the input tensor."
                        )

        return forward_pre_hook

    def _fused_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        """Forward via TE op-fuser (training only)."""
        if self._fused_ops is None:
            self._fused_ops = (self._make_fused_ops(),)
        (ops,) = self._fused_ops

        unpadded_tokens_per_expert = None
        _use_fp8 = getattr(self.config, "fp8", None)
        _use_fp4 = getattr(self.config, "fp4", None)
        if _skip_routed_expert_padding(self.config):
            pass
        elif _use_fp8 or _use_fp4:
            tokens_per_expert = tokens_per_expert.tolist()
            unpadded_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), unpadded_tokens_per_expert
            )
            permuted_probs = permuted_probs.squeeze(-1)
            tokens_per_expert = torch.tensor(
                tokens_per_expert, dtype=torch.int, device=permuted_probs.device
            )

        _moe_paged_stash = getattr(self.config, "moe_paged_stash", False)
        if _moe_paged_stash:
            permuted_local_hidden_states = paged_stash_group_start(permuted_local_hidden_states)
            max_num_tokens = permuted_local_hidden_states.shape[0]
            cap_factor = getattr(self.config, "moe_expert_rank_capacity_factor", None)
            avg_num_tokens = (
                int(max_num_tokens // cap_factor)
                if cap_factor is not None and cap_factor > 0
                else None
            )
            stash_context = get_paged_stash_context(
                name="grouped_mlp",
                max_num_tokens=max_num_tokens,
                num_tokens_tensor=tokens_per_expert.sum(),
                avg_num_tokens=avg_num_tokens,
            )
        else:
            stash_context = nullcontext()

        with stash_context:
            output = ops(
                permuted_local_hidden_states,
                tokens_per_expert,   # FC1
                permuted_probs,      # Scaled SwiGLU
                tokens_per_expert,   # FC2
            )

        if unpadded_tokens_per_expert is not None:
            output = self.quantization_unpadding(output, unpadded_tokens_per_expert)
        if _moe_paged_stash:
            output = paged_stash_group_commit(output, name="grouped_mlp")
        return output

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of TEGroupedMLP.

        Args:
            permuted_local_hidden_states: Input hidden states for local experts,
                shape ``[num_tokens, hidden_size]``.
            tokens_per_expert: Number of tokens routed to each expert,
                shape ``[num_local_experts]``.
            permuted_probs: Routing probabilities per token,
                shape ``[num_tokens]``.

        Returns:
            ``(output, output_bias)`` where ``output_bias`` is always ``None``
            (bias, if any, has already been fused into *output*).
        """
        # TE fused path
        if self._with_fused_impl:
            output = self._fused_forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
            return output, None

        _use_fp8 = getattr(self.config, "fp8", None)
        _use_fp4 = getattr(self.config, "fp4", None)

        # Apply quantization padding if needed
        unpadded_tokens_per_expert = None
        tokens_per_expert_list: list[int] = tokens_per_expert.tolist()
        permuted_probs = permuted_probs.unsqueeze(-1)

        if not _skip_routed_expert_padding(self.config) and (_use_fp8 or _use_fp4):
            unpadded_tokens_per_expert = tokens_per_expert_list
            permuted_local_hidden_states, tokens_per_expert_list = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert_list
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs, unpadded_tokens_per_expert
            )

        # Pre-multiply probs on input (single-expert-per-token optimisation)
        if getattr(self.config, "moe_apply_probs_on_input", False):
            assert self.config.moe_router_topk == 1, (
                "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            )
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        # FC1
        with off_interface(self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1") as phs:
            fc1_output, bias_parallel = self.linear_fc1(phs, tokens_per_expert_list)
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        # Activation (with optional recompute)
        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            with_glu_interleaving = (
                self.config.gated_linear_unit
                and getattr(self.config, "moe_mlp_glu_interleave_size", None) is not None
            )

            if getattr(self.config, "use_te_activation_func", False):
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if with_glu_interleaving:
                    intermediate_parallel = self._remove_glu_interleaving(
                        intermediate_parallel, self.config.moe_mlp_glu_interleave_size
                    )
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)

            elif getattr(self.config, "bias_activation_fusion", False) and not with_glu_interleaving:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        getattr(self.config, "activation_func_fp8_input_store", False),
                    )
                elif self.activation_func == _quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        getattr(self.config, "activation_func_fp8_input_store", False),
                        getattr(self.config, "glu_linear_offset", 0.0),
                        getattr(self.config, "activation_func_clamp_value", None),
                    )
                else:
                    raise ValueError(
                        "Only swiglu and quick_gelu support bias_activation_fusion in TEGroupedMLP."
                    )

            elif (
                self.activation_func == _squared_relu
                and getattr(self.config, "use_fused_weighted_squared_relu", False)
            ):
                assert bias_parallel is None, (
                    "Bias is not supported with fused weighted squared relu."
                )
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )

            else:
                if self.config.gated_linear_unit:
                    def glu(x):
                        if with_glu_interleaving:
                            x = self._remove_glu_interleaving(
                                x, self.config.moe_mlp_glu_interleave_size
                            )
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        clamp = getattr(self.config, "activation_func_clamp_value", None)
                        if clamp is not None:
                            x_glu = x_glu.clamp(min=None, max=clamp)
                            x_linear = x_linear.clamp(min=-clamp, max=clamp)
                        offset = getattr(self.config, "glu_linear_offset", 0.0)
                        return self.activation_func(x_glu) * (x_linear + offset)

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    if bias_parallel is not None:
                        intermediate_parallel = intermediate_parallel + bias_parallel
                    intermediate_parallel = self.activation_func(intermediate_parallel)

                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)

            return intermediate_parallel

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.activation_checkpoint.checkpoint(
                    bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)

        output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert_list)

        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        if self.offload_moe_act:
            output = off_interface.group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )

        output = self._apply_bias(output, output_bias, tokens_per_expert_list, permuted_probs)

        if unpadded_tokens_per_expert is not None:
            output = self.quantization_unpadding(output, unpadded_tokens_per_expert)

        output_bias = None
        return output, output_bias

    # ------------------------------------------------------------------
    # Checkpoint / sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> dict:
        """Map local expert weights to global expert shards.

        The layout is interchangeable with SequentialMLP's sharded_state_dict.
        """
        metadata = _ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
        sharded_state_dict: dict = {}

        for name, module in self._modules.items():
            sub_sd = _sharded_state_dict_default(
                module, f"{name}.", sharded_offsets, metadata, tp_group=self.tp_group
            )

            if name == "linear_fc1" and self.config.gated_linear_unit:
                ep_size = self.ep_group.size() if self.ep_group is not None else 1
                ep_rank = self.ep_group.rank() if self.ep_group is not None else 0
                num_global_experts = ep_size * self.num_local_experts
                local_expert_indices_offset = ep_rank * self.num_local_experts
                ep_axis = len(sharded_offsets)

                for i in range(self.num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f"{name}.weight{i}", f"{name}.bias{i}"):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )

            if singleton_local_shards:
                _replace_prefix_for_sharding(sub_sd, "", f"{prefix}experts.")
            else:
                _replace_prefix_for_sharding(sub_sd, f"{name}.", f"{prefix}experts.{name}.")

            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})

        return sharded_state_dict

    # ------------------------------------------------------------------
    # Deferred weight-gradient compute
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Backward pass for weight gradients.

        When ``delay_wgrad_compute`` is enabled via the TE fused path,
        fires the backward on the fused children in reverse order (FC2
        then FC1) and triggers DDP wgrad hooks on the original linear
        modules.
        """
        _delay = getattr(self.linear_fc1, "delay_wgrad_compute", False)
        if self._with_fused_impl and _delay:
            if self._fused_ops is not None:
                (seq,) = self._fused_ops
                fused_children = list(seq.children())
                assert len(fused_children) >= 3, "expected FC1, activation, FC2 in fused TE ops"
                fused_children[2].backward_dw()
                fused_children[0].backward_dw()
                self.linear_fc2._trigger_wgrad_accumulation_and_reduce_hooks()
                self.linear_fc1._trigger_wgrad_accumulation_and_reduce_hooks()
            return
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


# ---------------------------------------------------------------------------
# InferenceGroupedMLP — optional inference-optimised path
# ---------------------------------------------------------------------------

# FlashInfer fused MoE (optional)
try:
    import flashinfer.fused_moe as _flashinfer_fused_moe  # type: ignore[import]
    from flashinfer.fused_moe.core import ActivationType as _FlashInferActivationType  # type: ignore[import]
    _HAVE_FLASHINFER = True
except ImportError:
    _flashinfer_fused_moe = None  # type: ignore[assignment]
    _FlashInferActivationType = None  # type: ignore[assignment]
    _HAVE_FLASHINFER = False

# MXFP8 tensor (optional — requires sufficiently new Megatron/TE)
try:
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor as _MXFP8Tensor  # type: ignore[import]
    _HAVE_MXFP8 = True
except ImportError:
    try:
        from deepspeed.core.inference.quantization.mxfp8_tensor import MXFP8Tensor as _MXFP8Tensor  # type: ignore[import]
        _HAVE_MXFP8 = True
    except ImportError:
        _MXFP8Tensor = None  # type: ignore[assignment]
        _HAVE_MXFP8 = False

# Inference grouped-GEMM backend enum and fused-moe helpers (optional)
try:
    from deepspeed.core.inference.moe import (  # type: ignore[import]
        ActivationType as _McoreActivationType,
        InferenceGroupedGemmBackend,
        mcore_fused_moe,
        vllm_fused_moe,
    )
    _HAVE_INFERENCE_MOE = True
except ImportError:
    try:
        from megatron.core.inference.moe import (  # type: ignore[import]
            ActivationType as _McoreActivationType,
            InferenceGroupedGemmBackend,
            mcore_fused_moe,
            vllm_fused_moe,
        )
        _HAVE_INFERENCE_MOE = True
    except ImportError:
        _McoreActivationType = None  # type: ignore[assignment]
        InferenceGroupedGemmBackend = None  # type: ignore[assignment]
        mcore_fused_moe = None  # type: ignore[assignment]
        vllm_fused_moe = None  # type: ignore[assignment]
        _HAVE_INFERENCE_MOE = False

# Inference mode flag (optional)
try:
    from deepspeed.core.inference.utils import InferenceMode as _InferenceMode  # type: ignore[import]
    _HAVE_INFERENCE_MODE = True
except ImportError:
    try:
        from megatron.core.inference.utils import InferenceMode as _InferenceMode  # type: ignore[import]
        _HAVE_INFERENCE_MODE = True
    except ImportError:
        _InferenceMode = None  # type: ignore[assignment]
        _HAVE_INFERENCE_MODE = False

# NVLS all-gather dispatcher (optional)
try:
    from deepspeed.core.transformer.moe.token_dispatcher_inference import (  # type: ignore[import]
        InferenceAllGatherDispatcherBase as _InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher as _NVLSAllGatherVDispatcher,
    )
    _HAVE_NVLS_DISPATCHER = True
except ImportError:
    try:
        from megatron.core.transformer.moe.token_dispatcher_inference import (  # type: ignore[import]
            InferenceAllGatherDispatcherBase as _InferenceAllGatherDispatcherBase,
            NVLSAllGatherVDispatcher as _NVLSAllGatherVDispatcher,
        )
        _HAVE_NVLS_DISPATCHER = True
    except ImportError:
        _InferenceAllGatherDispatcherBase = None  # type: ignore[assignment]
        _NVLSAllGatherVDispatcher = None  # type: ignore[assignment]
        _HAVE_NVLS_DISPATCHER = False


class InferenceGroupedMLP(TEGroupedMLP):
    """Inference-optimised GroupedMLP with GPU-resident offsets.

    Inherits from TEGroupedMLP to reuse weight initialisation and checkpoint
    compatibility.  Supports three forward paths:

    * **Training / colocated RL**: delegates to parent ``TEGroupedMLP``.
    * **Inference + CUDA-graphed**: FlashInfer ``cutlass_fused_moe`` (fused
      permute + GEMM).  Requires ``flashinfer-python``.
    * **Inference + eager**: ``torch.nn.functional.grouped_mm`` with
      GPU-resident cumsum offsets via ``mcore_fused_moe`` or ``vllm_fused_moe``.

    When the optional inference backends (FlashInfer, mcore inference MoE) are
    not installed the module falls back to the parent TEGroupedMLP forward path
    so it can always be instantiated safely.

    Args:
        num_local_experts: Number of experts on this rank.
        config: TransformerConfig instance.
        submodules: GroupedMLPSubmodules with fc1/fc2 builders.
        pg_collection: Process-group bundle.
        name: Optional name for parameter naming.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
            name=name,
        )

        # Concatenated weight buffers are built lazily on first forward so that
        # checkpoint loading has a chance to populate per-expert parameters first.
        self._concatenated_weights_built = False

        # Resolve FlashInfer activation type (if FlashInfer is available)
        if _HAVE_FLASHINFER:
            self._flashinfer_activation_type = self._resolve_flashinfer_activation_type()
        else:
            self._flashinfer_activation_type = None

        # Resolve mcore inference activation type (if backend is available)
        if _HAVE_INFERENCE_MOE:
            try:
                self._mcore_activation_type = self._resolve_mcore_activation_type()
            except ValueError:
                self._mcore_activation_type = None
        else:
            self._mcore_activation_type = None

        self.inference_grouped_gemm_backend = getattr(
            config, "inference_grouped_gemm_backend", None
        )
        self._nvls_dispatcher = (
            getattr(config, "inference_moe_token_dispatcher_type", None) == "nvls"
        )

    # ------------------------------------------------------------------
    # Activation type resolution
    # ------------------------------------------------------------------

    def _resolve_flashinfer_activation_type(self):
        """Map config activation function to FlashInfer ActivationType."""
        assert _HAVE_FLASHINFER, "flashinfer-python is required."
        func = self.config.activation_func
        if func == F.silu:
            return _FlashInferActivationType.Silu
        if func == F.gelu:
            return _FlashInferActivationType.Gelu
        if func == F.relu:
            return _FlashInferActivationType.Relu
        if func == _squared_relu:
            return _FlashInferActivationType.Relu2
        raise ValueError(
            f"No FlashInfer ActivationType mapping for activation_func={func}"
        )

    def _resolve_mcore_activation_type(self):
        """Map config activation function to mcore_fused_moe ActivationType."""
        assert _HAVE_INFERENCE_MOE, "deepspeed.core.inference.moe is required."
        func = self.config.activation_func
        if func == _squared_relu:
            return _McoreActivationType.SQUARED_RELU
        raise ValueError(
            f"No mcore_fused_moe ActivationType mapping for activation_func={func}"
        )

    # ------------------------------------------------------------------
    # Concatenated weight builders
    # ------------------------------------------------------------------

    def _build_concatenated_mxfp8_weights(self):
        """Stack per-expert MXFP8Tensor attributes into contiguous buffers.

        After quantise_model_to_mxfp8, per-expert weights are MXFP8Tensors.
        This method stacks their ``.data`` and ``.scale`` into ``_fc1_weight``
        / ``_fc2_weight`` for ``scaled_grouped_mm``.

        Unlike ``_build_concatenated_weights`` this does **not** create
        ``nn.Parameter`` views — MXFP8 weights are plain attributes, not
        parameters.  Only used for non-colocated inference.
        """
        assert _HAVE_MXFP8, "MXFP8Tensor is required for MXFP8 weight concatenation."
        for linear_name, buf_name in [
            ("linear_fc1", "_fc1_weight"),
            ("linear_fc2", "_fc2_weight"),
        ]:
            linear = getattr(self, linear_name)
            q_list, s_list = [], []
            for i in range(self.num_local_experts):
                w = getattr(linear, f"weight{i}")
                if isinstance(w, _MXFP8Tensor):
                    mxfp8 = w
                elif hasattr(w, "data") and isinstance(w.data, _MXFP8Tensor):
                    mxfp8 = w.data
                else:
                    raise RuntimeError(
                        f"Expected MXFP8Tensor for {linear_name}.weight{i}, "
                        f"got {type(w).__name__}. Was quantize_model_to_mxfp8 called?"
                    )
                q_list.append(mxfp8.data)
                s_list.append(mxfp8.scale)

            stacked_data = torch.stack(q_list, dim=0).contiguous()
            stacked_scale = torch.stack(s_list, dim=0).contiguous()
            setattr(self, buf_name, _MXFP8Tensor(data=stacked_data, scale=stacked_scale))

            # Redirect per-expert weight views into the stacked buffer.
            for i in range(self.num_local_experts):
                w = getattr(linear, f"weight{i}")
                if isinstance(w, _MXFP8Tensor):
                    w.data = stacked_data[i]
                    w.scale = stacked_scale[i]
                elif hasattr(w, "data") and isinstance(w.data, _MXFP8Tensor):
                    w.data.data = stacked_data[i]
                    w.data.scale = stacked_scale[i]

    @torch.inference_mode(False)
    def _build_concatenated_weights(self):
        """Create contiguous weight tensors that share storage with TE's per-expert params.

        Creates ``_fc1_weight`` and ``_fc2_weight`` as contiguous tensors of
        shape ``[num_experts, out_features, in_features]``.  Each
        ``nn.Parameter``'s ``.data`` is redirected to be a view into the
        contiguous buffer so that:

        * TE's forward works correctly (same Parameter objects, same state).
        * Training updates flow through (param.data is a view into the buffer).
        * ``grouped_mm`` / FlashInfer can use the contiguous buffer directly.
        """
        device = self.linear_fc1.weight0.device
        dtype = self.linear_fc1.weight0.dtype

        fc1_shape = self.linear_fc1.weight0.shape  # [out_features, in_features]
        fc2_shape = self.linear_fc2.weight0.shape

        _fc1_weight = torch.empty(
            self.num_local_experts, *fc1_shape, device=device, dtype=dtype
        )
        _fc2_weight = torch.empty(
            self.num_local_experts, *fc2_shape, device=device, dtype=dtype
        )

        for i in range(self.num_local_experts):
            fc1_param = getattr(self.linear_fc1, f"weight{i}")
            fc2_param = getattr(self.linear_fc2, f"weight{i}")
            _fc1_weight[i].copy_(fc1_param.data)
            _fc2_weight[i].copy_(fc2_param.data)
            fc1_param.data = _fc1_weight[i]
            fc2_param.data = _fc2_weight[i]

        # Register as non-persistent buffers so .to(device) works but they are
        # not saved in state_dict (the views into them are saved via parameters).
        self.register_buffer("_fc1_weight", _fc1_weight, persistent=False)
        self.register_buffer("_fc2_weight", _fc2_weight, persistent=False)

    # ------------------------------------------------------------------
    # Inference forward paths
    # ------------------------------------------------------------------

    def _flashinfer_forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FlashInfer fused MoE kernel (CUDA-graphed inference)."""
        assert _HAVE_FLASHINFER, "flashinfer-python is required for FlashInfer forward."
        assert probs.dtype == torch.float32, (
            "FlashInfer forward path requires fp32 probabilities."
        )
        out_tensor = None
        if self._nvls_dispatcher and _HAVE_NVLS_DISPATCHER:
            out_tensor = _NVLSAllGatherVDispatcher._get_rsv_tensor()
        output = _flashinfer_fused_moe.cutlass_fused_moe(
            hidden_states,
            routing_map.int(),
            probs,
            self._fc1_weight,
            self._fc2_weight,
            hidden_states.dtype,
            quant_scales=None,
            activation_type=self._flashinfer_activation_type,
            ep_size=self.ep_group.size(),
            ep_rank=self.ep_group.rank(),
            output=out_tensor,
        )[0]
        return output, None

    def _mcore_fused_moe_forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Torch grouped_mm fused MoE forward via mcore_fused_moe."""
        assert _HAVE_INFERENCE_MOE, "deepspeed.core.inference.moe is required."
        local_expert_start = self.ep_group.rank() * self.num_local_experts
        valid_tokens = None
        out_tensor = None
        if _HAVE_NVLS_DISPATCHER:
            if _InferenceAllGatherDispatcherBase is not None:
                valid_tokens = _InferenceAllGatherDispatcherBase._valid_tokens()
            if self._nvls_dispatcher and _NVLSAllGatherVDispatcher is not None:
                out_tensor = _NVLSAllGatherVDispatcher._get_rsv_tensor()
        output = mcore_fused_moe(
            hidden_states,
            probs,
            self._fc1_weight,
            self._fc2_weight,
            activation_type=self._mcore_activation_type,
            num_local_experts=self.num_local_experts,
            local_expert_start=local_expert_start,
            valid_tokens=valid_tokens,
            routing_map=routing_map,
            disable_fused_quant_kernels=getattr(
                self.config, "inference_moe_disable_fused_quant_kernels", False
            ),
            out=out_tensor,
        )
        return output, None

    def _vllm_forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """vLLM Triton fused MoE kernel forward (BF16, CUDA-graph safe)."""
        assert _HAVE_INFERENCE_MOE, "deepspeed.core.inference.moe is required."
        local_expert_start = self.ep_group.rank() * self.num_local_experts
        valid_tokens = None
        out_tensor = None
        num_tokens_hint = None
        if _HAVE_NVLS_DISPATCHER:
            if _InferenceAllGatherDispatcherBase is not None:
                valid_tokens = _InferenceAllGatherDispatcherBase._valid_tokens()
                num_tokens_hint = (
                    _InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate()
                )
            if self._nvls_dispatcher and _NVLSAllGatherVDispatcher is not None:
                out_tensor = _NVLSAllGatherVDispatcher._get_rsv_tensor()
        output = vllm_fused_moe(
            hidden_states,
            probs,
            self._fc1_weight,
            self._fc2_weight,
            activation_type=self._mcore_activation_type,
            num_local_experts=self.num_local_experts,
            local_expert_start=local_expert_start,
            valid_tokens=valid_tokens,
            routing_map=routing_map,
            out=out_tensor,
            num_tokens_hint=num_tokens_hint,
        )
        return output, None

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor],
        permuted_probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with three modes.

        Args:
            permuted_local_hidden_states: ``[num_tokens, hidden_size]`` input
                hidden states.
            tokens_per_expert: ``[num_experts]`` token counts per expert.
                ``None`` when using the CUDA-graphed FlashInfer path.
            permuted_probs: ``[num_tokens, topk]`` routing probabilities.
            routing_map: ``[num_tokens, topk]`` token-to-expert assignment
                indices.  Required for the FlashInfer CUDA-graphed path,
                ``None`` otherwise.

        Returns:
            ``(output, None)`` — output bias is always ``None`` in grouped
            inference paths.
        """
        # In training / colocated-RL mode fall back to parent TEGroupedMLP.
        inference_active = (
            _HAVE_INFERENCE_MODE and _InferenceMode is not None and _InferenceMode.is_active()
        )
        if not inference_active:
            fp8_recipe = getattr(self.config, "fp8_recipe", None)
            assert fp8_recipe != "mxfp8", (
                "MXFP8 inference-optimised path is not compatible with training / colocated RL."
            )
            return super().forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

        # Lazily build concatenated weight buffers after checkpoint load.
        if not self._concatenated_weights_built:
            w = getattr(self.linear_fc1, "weight0", None)
            if (
                _HAVE_MXFP8
                and w is not None
                and (
                    isinstance(w, _MXFP8Tensor)
                    or (hasattr(w, "data") and isinstance(w.data, _MXFP8Tensor))
                )
            ):
                self._build_concatenated_mxfp8_weights()
            else:
                self._build_concatenated_weights()
            self._concatenated_weights_built = True

        # Dispatch to the appropriate inference kernel.
        if (
            _HAVE_INFERENCE_MOE
            and InferenceGroupedGemmBackend is not None
            and self.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.FLASHINFER
        ):
            assert routing_map is not None, (
                "routing_map is required for the FlashInfer forward pass."
            )
            assert not self.training, (
                "FlashInfer forward path is only used in inference mode."
            )
            return self._flashinfer_forward(
                permuted_local_hidden_states, routing_map, permuted_probs
            )
        elif (
            _HAVE_INFERENCE_MOE
            and InferenceGroupedGemmBackend is not None
            and self.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.TORCH
        ):
            return self._mcore_fused_moe_forward(
                permuted_local_hidden_states, permuted_probs, routing_map=routing_map
            )
        elif (
            _HAVE_INFERENCE_MOE
            and InferenceGroupedGemmBackend is not None
            and self.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.VLLM
        ):
            return self._vllm_forward(
                permuted_local_hidden_states, permuted_probs, routing_map=routing_map
            )
        else:
            # No inference backend configured — fall back to eager TEGroupedMLP path.
            assert tokens_per_expert is not None, (
                "tokens_per_expert is required when no inference grouped-GEMM backend is set."
            )
            return super().forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )


# ---------------------------------------------------------------------------
# SequentialMLP
# ---------------------------------------------------------------------------

class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    Executes each expert sequentially.  Simpler than TEGroupedMLP but less
    efficient for large expert counts.

    Args:
        num_local_experts: Number of experts local to this rank.
        config: TransformerConfig instance.
        submodules: MLPSubmodules for per-expert MLP construction.
        pg_collection: Process-group bundle (ep, expt_tp, expt_dp groups).
        name: Optional module name for parameter naming.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        pg_collection = pg_collection or ProcessGroupCollection()

        moe_ffn_hidden_size = getattr(config, "moe_ffn_hidden_size", None)
        ffn_hidden_size = getattr(config, "ffn_hidden_size", None) or config.hidden_size * 4

        if moe_ffn_hidden_size == ffn_hidden_size or moe_ffn_hidden_size is None:
            super().__init__(config=config)
        else:
            # Override ffn_hidden_size with moe_ffn_hidden_size via a deepcopy
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.dp_group = getattr(pg_collection, "expt_dp", None)

        _moe_ffn = getattr(self.config, "moe_ffn_hidden_size", None)

        for expert_idx in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=_moe_ffn,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
                name=(name + f".local_experts.{expert_idx}") if name is not None else None,
            )
            self.local_experts.append(expert)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_tensor_for_quantization(
        self, hidden: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad token dimension to multiples of 16 (FP8) or 32 (FP4)."""
        actual_num_tokens = hidden.shape[0]
        divisor = _get_align_size(self.config)
        pad_count = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if pad_count > 0:
            pad_h = torch.zeros(pad_count, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
            hidden = torch.cat((hidden, pad_h), dim=0)
            pad_p = torch.zeros(pad_count, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_p), dim=0)
        return hidden, probs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward step of the SequentialMLP.

        Args:
            permuted_local_hidden_states: Permuted hidden states,
                shape ``[num_tokens, hidden_size]``.
            tokens_per_expert: Tokens per expert, shape ``[num_local_experts]``.
            permuted_probs: Routing probs per token, shape ``[num_tokens]``.

        Returns:
            ``(output, output_bias)`` — output_bias is always ``None`` since
            expert bias is already added inside each MLP.
        """
        _use_fp8 = getattr(self.config, "fp8", None)
        _use_fp4 = getattr(self.config, "fp4", None)

        # Pre-apply routing probs on input (top-1 only)
        if getattr(self.config, "moe_apply_probs_on_input", False):
            assert self.config.moe_router_topk == 1, (
                "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            )
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if _use_fp8 or _use_fp4:
                hidden, probs = self._pad_tensor_for_quantization(
                    permuted_local_hidden_states, permuted_probs
                )
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](
                    permuted_local_hidden_states, permuted_probs
                )
            return output, output_bias

        # Multiple experts
        tokens_per_expert_list: list[int] = tokens_per_expert.tolist()
        tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert_list)
        probs_list = torch.split(permuted_probs, tokens_per_expert_list)

        output_local_list = []
        for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
            if _use_fp8 or _use_fp4:
                hidden, probs_pad = self._pad_tensor_for_quantization(tokens, probs)
                output, _ = expert(hidden, probs_pad)
                output = output[: tokens.shape[0]]
            else:
                output, _ = expert(tokens, probs)
            output_local_list.append(output)

        output_local = torch.cat(output_local_list, dim=0)
        return output_local, None

    # ------------------------------------------------------------------
    # Deferred weight-gradient compute
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Backward pass for weight gradients across all local experts."""
        for expert in self.local_experts:
            expert.backward_dw()

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> dict:
        """Map local expert weights to global expert shards."""
        metadata = _ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict: dict = {}
        ep_size = self.ep_group.size() if self.ep_group is not None else 1
        ep_rank = self.ep_group.rank() if self.ep_group is not None else 0
        dp_rank = self.dp_group.rank() if self.dp_group is not None else 0
        num_global_experts = ep_size * self.num_local_experts
        local_expert_indices_offset = ep_rank * self.num_local_experts
        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f"{prefix}local_experts.{expert_local_idx}."

            if singleton_local_shards:
                expert_sharded_prefix = f"{prefix}experts.{expert_global_idx}."
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f"{prefix}experts."
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            expert_state_dict = _sharded_state_dict_default(
                expert, expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            _replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )

            # Adjust replica ids — replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                if isinstance(sh_ten, dict) and "replica_id" in sh_ten:
                    replica_id = sh_ten["replica_id"]
                    assert len(replica_id) == 3, (
                        f"Expected replica_id for {k} to be in (PP, TP, DP) format, "
                        f"got: {replica_id}"
                    )
                    sh_ten["replica_id"] = (*replica_id[:2], dp_rank)

            sharded_state_dict.update(expert_state_dict)

        return sharded_state_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GroupedMLPSubmodules",
    "GroupedLinearFc1Interface",
    "GroupedLinearFc1Builder",
    "GroupedLinearFc2Interface",
    "GroupedLinearFc2Builder",
    "TEGroupedMLP",
    "InferenceGroupedMLP",
    "SequentialMLP",
]
