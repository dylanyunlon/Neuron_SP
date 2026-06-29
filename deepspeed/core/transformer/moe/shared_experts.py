# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""SharedExpertMLP — shared (non-routed) expert for Mixture-of-Experts.

Ported from Megatron-LM megatron/core/transformer/moe/shared_experts.py.

Key differences vs. Megatron original
--------------------------------------
* All ``megatron.core.*`` imports replaced with ``deepspeed.core.*`` equivalents.
* ``HAVE_TE`` / ``TELinear`` / ``set_save_original_input`` sourced from
  ``deepspeed.core.transformer.attention`` (same lazy-import pattern used there).
* ``ProcessGroupCollection`` comes from ``deepspeed.core.process_groups_config``
  instead of ``megatron.core.transformer.moe.moe_utils``.
* ``apply_module`` is not used in DeepSpeed; direct ``self.linear_fc*(…)`` calls
  are used, matching how ``mlp.py`` invokes its linear layers.
* ``make_sharded_tensor_for_checkpoint`` / ``ShardedStateDict`` replaced by the
  plain-dict scheme used throughout ``deepspeed.core.transformer.module``.
* ``is_te_min_version`` / ``is_torch_min_version`` are emulated with simple
  ``packaging.version`` comparisons (or no-ops when packaging is absent).
* The parallel communication functions (AG / RS for SP, copy / reduce for TP)
  are imported from ``deepspeed.core.tensor_parallel.mappings``; note that the
  DeepSpeed versions do **not** accept a ``group`` keyword argument — the global
  parallel state group is used automatically (consistent with token_dispatcher.py).
* ``moe_shared_expert_overlap`` path is fully preserved.  The stream-based
  overlap methods (``pre_forward_comm``, ``linear_fc1_forward_and_act``,
  ``linear_fc2_forward``, ``post_forward_comm``, ``get_output``) are intact.
* ``set_tensor_grad_fn_sequence_sr`` helper kept as-is (PyTorch API is the same).
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DeepSpeed-local imports (replacing megatron.core.*)
# ---------------------------------------------------------------------------

from deepspeed.core.dist_checkpointing.mapping import ShardedStateDict
from deepspeed.core.process_groups_config import ProcessGroupCollection
from deepspeed.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from deepspeed.core.transformer.mlp import MLP, MLPSubmodules
from deepspeed.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Transformer Engine helpers (optional dependency)
# ---------------------------------------------------------------------------
# Mirror the lazy-import pattern from deepspeed/core/transformer/attention.py
try:
    import transformer_engine  # noqa: F401  (existence check)
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if HAVE_TE:
    try:
        from transformer_engine.pytorch import Linear as TELinear
        from transformer_engine.pytorch.fp8 import set_save_original_input  # type: ignore[import]
    except Exception:  # TE installed but API differs
        TELinear = None  # type: ignore[assignment]
        set_save_original_input = None  # type: ignore[assignment]
else:
    TELinear = None  # type: ignore[assignment]
    set_save_original_input = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Version-check helpers
# ---------------------------------------------------------------------------

def _parse_version(version_str: str) -> tuple:
    """Return a comparable tuple from a version string like '2.6.0dev0'."""
    import re
    nums = re.findall(r"\d+", version_str)
    return tuple(int(n) for n in nums)


def is_te_min_version(min_version: str) -> bool:
    """Return True if Transformer Engine >= *min_version* is installed."""
    if not HAVE_TE:
        return False
    try:
        import transformer_engine
        return _parse_version(transformer_engine.__version__) >= _parse_version(min_version)
    except Exception:
        return False


def is_torch_min_version(min_version: str) -> bool:
    """Return True if PyTorch >= *min_version* is installed."""
    try:
        return _parse_version(torch.__version__) >= _parse_version(min_version)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class SharedExpertState(Enum):
    """State machine states for SharedExpertMLP overlapped forward pass."""

    IDLE = 0
    PRE_FORWARD_COMM_DONE = 1
    FC1_FORWARD_DONE = 2
    FC2_FORWARD_DONE = 3
    POST_FORWARD_COMM_DONE = 4


def overlap_state_check(required_state: "SharedExpertState", next_state: "SharedExpertState"):
    """Decorator to validate overlap state and cached variables before method execution,
    and update state after method execution.

    Args:
        required_state: The expected SharedExpertState before this method runs.
        next_state: The SharedExpertState to transition to after method execution.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Check overlap is enabled
            assert (
                self.config.moe_shared_expert_overlap
            ), f"{method.__name__} requires moe_shared_expert_overlap to be set in config"
            # Check state machine
            assert self._overlap_state == required_state, (
                f"{method.__name__} must be called from {required_state.name} state, "
                f"but current state is {self._overlap_state.name}"
            )
            # Execute method
            result = method(self, *args, **kwargs)
            # Update state after method execution
            self._overlap_state = next_state
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Backward stream synchronisation helper
# ---------------------------------------------------------------------------

class _BackwardStreamWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, stream):
        """forward"""
        ctx.stream = stream
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """backward with stream wait"""
        ctx.stream.wait_stream(torch.cuda.current_stream())
        return grad_output, None


# ---------------------------------------------------------------------------
# SharedExpertMLP
# ---------------------------------------------------------------------------

class SharedExpertMLP(MLP):
    """MLP layer for Shared Experts.

    Extends :class:`deepspeed.core.transformer.mlp.MLP` with:

    * Optional gating via a learned scalar weight (``use_shared_expert_gate``).
    * Overlapped forward pass split into staged methods when
      ``config.moe_shared_expert_overlap`` is True.

    The overlapped forward pass must be called in this exact order::

        shared_expert.pre_forward_comm(input)
        shared_expert.linear_fc1_forward_and_act()
        shared_expert.linear_fc2_forward()
        shared_expert.post_forward_comm()
        output = shared_expert.get_output()

    When overlap is disabled, use the standard ``forward(hidden_states)``
    inherited path augmented by the gate.

    Args:
        config: TransformerConfig instance.
        submodules: MLPSubmodules specifying fc1 / fc2 builders.
        gate: Whether to use a shared-expert gate (sigmoid scalar weight).
        pg_collection: Optional ProcessGroupCollection; its ``.tp`` group is
            forwarded to the MLP base class for tensor-parallel communication.
        name: Optional module name passed top-down from the parent module.
    """

    # Class-level CUDA stream shared across all instances (set on first use).
    stream: Optional[torch.cuda.Stream] = None  # type: ignore[type-arg]

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: Optional[str] = None,
    ) -> None:
        config = deepcopy(config)
        assert config.add_bias_linear is False, (
            "Bias is not supported in the shared experts; "
            "please set add_bias_linear=False in TransformerConfig."
        )

        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size

        tp_group = pg_collection.tp if pg_collection is not None else None
        super().__init__(
            config=config,
            submodules=submodules,
            tp_group=tp_group,
            name=name,
        )

        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            self.gate_weight = torch.nn.Parameter(
                torch.empty((1, self.config.hidden_size))
            )
            perform_initialization = getattr(config, "perform_initialization", True)
            if perform_initialization:
                config.init_method(self.gate_weight)
            params_dtype = getattr(config, "params_dtype", None)
            if params_dtype is not None:
                self.gate_weight.data = self.gate_weight.data.to(dtype=params_dtype)
            sequence_parallel = getattr(self.config, "sequence_parallel", False)
            setattr(self.gate_weight, "sequence_parallel", sequence_parallel)
        else:
            self.gate_weight = None

        # FP8 / FP4: avoid storing redundant quantised copy of pre-layernorm output.
        fp8 = getattr(config, "fp8", None)
        fp4 = getattr(config, "fp4", None)
        fp8_recipe = getattr(config, "fp8_recipe", None)
        if (
            fp8 and fp8_recipe != "delayed" and is_te_min_version("2.6.0dev0")
        ) or (fp4 and is_te_min_version("2.7.0.dev0")):
            recompute_granularity = getattr(config, "recompute_granularity", None)
            recompute_modules = getattr(config, "recompute_modules", None) or []
            shared_experts_recompute = (
                recompute_granularity == "selective"
                and "shared_experts" in recompute_modules
            )
            if (
                not shared_experts_recompute
                and HAVE_TE
                and TELinear is not None
                and set_save_original_input is not None
                and isinstance(self.linear_fc1, TELinear)
            ):
                set_save_original_input(self.linear_fc1)

        if self.config.moe_shared_expert_overlap:
            # Disable TP-related AG/RS communications in the linear modules so
            # that the staged overlap methods can inject them at the right points.
            for linear in [self.linear_fc1, self.linear_fc2]:
                if hasattr(linear, "parallel_mode"):
                    # TELinear
                    linear.parallel_mode = None
                    linear.ub_overlap_rs_fprop = False
                    linear.ub_overlap_ag_dgrad = False
                    linear.ub_overlap_ag_fprop = False
                    linear.ub_overlap_rs_dgrad = False
                else:
                    # MCore-style native Linear
                    linear.explicit_expert_comm = True

            # Cached intermediate tensors for the staged overlap path.
            self.cached_fc1_input: Optional[torch.Tensor] = None
            self.cached_fc2_input: Optional[torch.Tensor] = None
            self.cached_fc2_output: Optional[torch.Tensor] = None
            self.cached_output: Optional[torch.Tensor] = None
            self.gate_score: Optional[torch.Tensor] = None

            # State machine to ensure correct calling order.
            self._overlap_state = SharedExpertState.IDLE

            if self.__class__.stream is None:
                self.__class__.stream = torch.cuda.Stream()
            self.stream: torch.cuda.Stream = self.__class__.stream  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Standard forward (non-overlapped path)
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward function (non-overlapped path)."""
        output, _ = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Return sharded state dict, including gate_weight when present."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.use_shared_expert_gate:
            name = "gate_weight"
            state_dict = self.state_dict(prefix="", keep_vars=True)
            param = state_dict[name]
            key = f"{prefix}{name}"
            entry: dict = {"param": param, "shape": tuple(param.shape)}
            if getattr(param, "tensor_model_parallel", False):
                entry["tp_shard"] = {
                    "dim": getattr(param, "partition_dim", 0),
                    "stride": getattr(param, "partition_stride", 1),
                }
            if sharded_offsets:
                entry["sharded_offsets"] = sharded_offsets
            sharded_state_dict[key] = entry
        return sharded_state_dict

    # ------------------------------------------------------------------
    # Overlapped forward helpers
    # ------------------------------------------------------------------

    def wait_current_stream(self) -> None:
        """Make the shared-expert CUDA stream wait for the current stream."""
        self.stream.wait_stream(torch.cuda.current_stream())

    @overlap_state_check(SharedExpertState.IDLE, SharedExpertState.PRE_FORWARD_COMM_DONE)
    def pre_forward_comm(self, input: torch.Tensor, wait_current_stream: bool = True) -> None:
        """All-Gather for SP before forward (overlap path).

        Gathers the input tensor across the sequence-parallel region (or copies
        it into the TP region) so that the subsequent fc1 forward sees the full
        sequence.  Also pre-computes the gate score when a gate weight is used.

        Only effective when ``config.moe_shared_expert_overlap`` is True.

        Args:
            input: Hidden states tensor from the current SP rank.
            wait_current_stream: If True, synchronise the shared stream with
                the current CUDA stream before launching ops.
        """
        if wait_current_stream:
            self.wait_current_stream()
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                logits = torch.nn.functional.linear(input, self.gate_weight)
                self.gate_score = torch.nn.functional.sigmoid(logits)
            sequence_parallel = getattr(self.config, "sequence_parallel", False)
            if sequence_parallel:
                self.cached_fc1_input = gather_from_sequence_parallel_region(
                    input, tensor_parallel_output_grad=True, group=self.tp_group  # M3981
                )
            else:
                self.cached_fc1_input = copy_to_tensor_model_parallel_region(
                    input, group=self.tp_group  # M3981
                )
            set_tensor_grad_fn_sequence_sr(self.cached_fc1_input, torch.iinfo(torch.int).max)

    @overlap_state_check(
        SharedExpertState.PRE_FORWARD_COMM_DONE, SharedExpertState.FC1_FORWARD_DONE
    )
    def linear_fc1_forward_and_act(
        self, overlapped_comm_output: Optional[torch.Tensor] = None
    ) -> None:
        """FC1 linear + activation forward (overlap path).

        Runs ``linear_fc1`` and the configured activation function on the
        cached gathered input, storing the result for the next stage.

        Only effective when ``config.moe_shared_expert_overlap`` is True.

        Args:
            overlapped_comm_output: Optional tensor whose ``grad_fn`` sequence
                number is used to control backward ordering relative to the
                routed expert dispatch communication.
        """
        with torch.cuda.stream(self.stream):
            # [s, b, 4 * h/p]
            intermediate_parallel, bias_parallel = self.linear_fc1(self.cached_fc1_input)
            self.cached_fc1_input = None

            # Activation (mirrors MLP._apply_activation logic)
            intermediate_parallel = self._apply_activation(
                intermediate_parallel, bias_parallel
            )

            self.cached_fc2_input = intermediate_parallel

        # Tensor sequence number is used to control the backward order.
        # Decrease the sequence number of the expert output to make the comm
        # launched first in the backward order.
        if overlapped_comm_output is not None and overlapped_comm_output.grad_fn is not None:
            target_sequence_nr = overlapped_comm_output.grad_fn._sequence_nr() - 1
            set_tensor_grad_fn_sequence_sr(intermediate_parallel, target_sequence_nr)
            # Make sure the shared expert fc1 backward is launched after the
            # routed fc1 backward.
            self.cached_fc2_input = _BackwardStreamWait.apply(
                intermediate_parallel, self.stream
            )

    @overlap_state_check(SharedExpertState.FC1_FORWARD_DONE, SharedExpertState.FC2_FORWARD_DONE)
    def linear_fc2_forward(
        self, overlapped_comm_output: Optional[torch.Tensor] = None
    ) -> None:
        """FC2 linear forward (overlap path).

        Only effective when ``config.moe_shared_expert_overlap`` is True.

        Args:
            overlapped_comm_output: Optional tensor whose sequence number is
                bumped to max so that it is scheduled last in backward.
        """
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.stream):
            # [s, b, h]
            self.cached_fc2_output, _ = self.linear_fc2(self.cached_fc2_input)
            self.cached_fc2_input = None

    @overlap_state_check(
        SharedExpertState.FC2_FORWARD_DONE, SharedExpertState.POST_FORWARD_COMM_DONE
    )
    def post_forward_comm(self) -> None:
        """Reduce-scatter for SP after forward (overlap path).

        Only effective when ``config.moe_shared_expert_overlap`` is True.
        """
        with torch.cuda.stream(self.stream):
            sequence_parallel = getattr(self.config, "sequence_parallel", False)
            if sequence_parallel:
                self.cached_output = reduce_scatter_to_sequence_parallel_region(
                    self.cached_fc2_output
                )
            else:
                self.cached_output = reduce_from_tensor_model_parallel_region(
                    self.cached_fc2_output
                )
            self.cached_fc2_output = None
            set_tensor_grad_fn_sequence_sr(self.cached_output, torch.iinfo(torch.int).max)

    @overlap_state_check(SharedExpertState.POST_FORWARD_COMM_DONE, SharedExpertState.IDLE)
    def get_output(self) -> torch.Tensor:
        """Retrieve the final output after all overlap stages (overlap path).

        Only effective when ``config.moe_shared_expert_overlap`` is True.

        Returns:
            Output tensor ``[s, b, h]`` (same sequence-parallel sharding as
            the original input to ``pre_forward_comm``).
        """
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                assert self.gate_score is not None
                output = self.cached_output * self.gate_score
                self.gate_score = None
            else:
                output = self.cached_output
            self.cached_output = None
        torch.cuda.current_stream().wait_stream(self.stream)
        return output


# ---------------------------------------------------------------------------
# Backward-order control helper
# ---------------------------------------------------------------------------

def set_tensor_grad_fn_sequence_sr(tensor: Optional[torch.Tensor], value: int) -> None:
    """Set the ``_sequence_nr`` of a tensor's ``grad_fn`` to *value*.

    Controls the backward scheduling order: a higher value causes the
    ``grad_fn`` to be scheduled earlier.  For PyTorch < 2.2.0 this is a no-op
    and a warning is emitted.

    Args:
        tensor: The tensor whose ``grad_fn`` sequence number to adjust.
            No-op when *tensor* is None or has no ``grad_fn``.
        value: The new sequence number.  Use ``torch.iinfo(torch.int).max``
            to schedule as early as possible, or a smaller value to defer.
    """
    if is_torch_min_version("2.2.0"):
        if tensor is not None and tensor.grad_fn is not None:
            tensor.grad_fn._set_sequence_nr(value)
    else:
        warnings.warn(
            "WARNING: PyTorch is too old to set sequence_nr and the performance may not "
            "be optimal. Please use PyTorch >= 2.2.0 for better performance."
        )
