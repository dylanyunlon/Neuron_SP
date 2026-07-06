"""Fused Layer and RMS Normalisation.

Mirrors Megatron-LM megatron/core/fusions/fused_layer_norm.py.

Provides four public classes and one factory function:

``FusedLayerNorm``
    LayerNorm with three-tier dispatch:
    1. ``apex.contrib.layer_norm.FastLayerNormFN`` — persistent CUDA kernel,
       requires ``hidden_size`` in the allow-list *and* apex installed.
    2. ``apex.normalization.FusedLayerNormAffineFunction`` — apex fused LN,
       works for any hidden size when apex is present.
    3. ``torch.nn.functional.layer_norm`` — pure-PyTorch fallback; always
       available, suitable for SM86 (A6000) on the DES-LOC cluster where apex
       may not be compiled.

``FusedRMSNorm``
    Root-Mean-Square LayerNorm (no mean subtraction; used in Llama / Mistral).
    Three-tier dispatch mirrors FusedLayerNorm:
    1. ``apex.normalization.FusedRMSNormAffineFunction`` (apex ≥ 23.08).
    2. ``apex.normalization.FusedRMSNormFunction`` (no affine transform).
    3. Pure-PyTorch rsqrt fallback.

``MixedFusedLayerNorm``
    Convenience subclass of ``FusedLayerNorm`` that accepts
    ``normalized_shape`` as a positional argument matching the
    ``torch.nn.LayerNorm`` constructor, making it a drop-in replacement
    without requiring a full ``TransformerConfig``.

``MixedFusedRMSNorm``
    Same convenience wrapper around ``FusedRMSNorm``.

``get_layer_norm(config)``
    Factory that returns the appropriate class (*not* an instance) for
    the normalisation type specified in ``config.normalization``:
    - ``"LayerNorm"``  → ``FusedLayerNorm``
    - ``"RMSNorm"``   → ``FusedRMSNorm``

All classes are used by ``TESpecProvider.layer_norm`` when TE < 1.9 or for
non-QK layer norms and by ``TransformerBlock`` directly.

Megatron source: Megatron-LM/megatron/core/fusions/fused_layer_norm.py
"""
from __future__ import annotations

import importlib
import inspect
import numbers

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.utils import make_viewless_tensor


# ---------------------------------------------------------------------------
# Optional apex imports — graceful fallback when apex is unavailable.
# ---------------------------------------------------------------------------

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN  # type: ignore

    HAVE_PERSIST_LAYER_NORM = True
except ImportError:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from apex.normalization.fused_layer_norm import (  # type: ignore
        FusedLayerNormAffineFunction,
    )

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    HAVE_FUSED_LAYER_NORM = False

# RMSNorm variants — available in apex ≥ 23.08.
try:
    from apex.normalization.fused_layer_norm import (  # type: ignore
        FusedRMSNormAffineFunction,
    )

    HAVE_FUSED_RMS_NORM_AFFINE = True
except ImportError:
    HAVE_FUSED_RMS_NORM_AFFINE = False

try:
    from apex.normalization.fused_layer_norm import (  # type: ignore
        FusedRMSNormFunction,
    )

    HAVE_FUSED_RMS_NORM = True
except ImportError:
    HAVE_FUSED_RMS_NORM = False


class FusedLayerNorm(torch.nn.Module):
    """Layer Norm fused into a single CUDA kernel when apex is available.

    Falls back to ``torch.nn.functional.layer_norm`` on clusters without
    apex (e.g. the DES-LOC A6000 nodes), so training is always functional.

    Args:
        config: ``TransformerConfig`` containing ``normalization``,
            ``layernorm_zero_centered_gamma``, ``persist_layer_norm``,
            ``memory_efficient_layer_norm``, and ``sequence_parallel``.
        hidden_size: Transformer hidden dimension.
        eps: Epsilon for numerical stability.
        persist_layer_norm: Prefer the persistent FastLayerNorm kernel.
            Automatically disabled when apex is absent or ``hidden_size``
            is not in the allow-list.
        zero_centered_gamma: Initialise weight to 0 (effective gamma = 1).
        normalization: Must be ``"LayerNorm"`` (included to match TE interface).
    """

    # Hidden sizes for which the apex persistent kernel is available.
    _PERSIST_LN_HIDDEN_SIZES = {
        1024, 1536, 2048, 2304, 3072, 3840, 4096, 5120, 6144, 8192,
        10240, 12288, 12800, 15360, 16384, 18432, 20480, 24576, 25600,
        30720, 32768, 40960, 49152, 65536,
    }

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ) -> None:
        super().__init__()

        self.config = config

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma
        assert self.config.normalization == "LayerNorm", (
            f"FusedLayerNorm only supports 'LayerNorm', got '{self.config.normalization}'"
        )

        # Resolve effective persist flag: need apex AND supported size.
        use_persist = self.config.persist_layer_norm
        if hidden_size not in self._PERSIST_LN_HIDDEN_SIZES or not HAVE_PERSIST_LAYER_NORM:
            use_persist = False
        self.persist_layer_norm = use_persist

        # Normalise hidden_size to a torch.Size for apex compatibility.
        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps

        # Parameters are initialised with torch.empty for correct device
        # placement under NeMo 2 / device-mesh workflows.
        self.weight = Parameter(torch.empty(*hidden_size))
        self.bias = Parameter(torch.empty(*hidden_size))
        self.reset_parameters()

        # Sequence parallelism flag propagated to weight/bias for TP sharding.
        self.sequence_parallel = getattr(self.config, "sequence_parallel", False)
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.bias, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self) -> None:
        """Initialise weight and bias.

        With ``zero_centered_gamma`` the weight is initialised to zero
        (effective gamma = weight + 1 = 1) for better numerical stability
        during training with large learning rates.
        """
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Apply layer normalisation.

        Dispatches to the apex persistent kernel, the apex fused kernel, or
        PyTorch's native layer_norm in order of preference.

        Args:
            input: Arbitrary-shape tensor; normalisation is over the last
                ``len(self.hidden_size)`` dimensions.

        Returns:
            Normalised tensor, same shape and dtype as ``input``.
        """
        # Zero-centred gamma: shift weight so the initialised value equals 1.
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if self.persist_layer_norm:
            # apex.contrib persistent fast layer norm
            fwd_args = inspect.getfullargspec(FastLayerNormFN.forward).args
            if "memory_efficient" in fwd_args:
                output = FastLayerNormFN.apply(
                    input, weight, self.bias, self.eps,
                    self.config.memory_efficient_layer_norm,
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex may return a view tensor; wrap in a viewless tensor so that
            # pipeline-parallel schedule.py::deallocate_output_tensor() does not
            # trip on a populated _base field.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )
            return output

        if HAVE_FUSED_LAYER_NORM:
            # apex.normalization fused layer norm (any hidden size)
            fwd_args = inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args
            if "memory_efficient" in fwd_args:
                return FusedLayerNormAffineFunction.apply(
                    input, weight, self.bias, self.hidden_size, self.eps,
                    self.config.memory_efficient_layer_norm,
                )
            else:
                return FusedLayerNormAffineFunction.apply(
                    input, weight, self.bias, self.hidden_size, self.eps,
                )

        # Pure-PyTorch fallback — always correct, slightly lower throughput.
        # Suitable for A6000 nodes where apex is not compiled.
        return torch.nn.functional.layer_norm(
            input, list(self.hidden_size), weight, self.bias, self.eps
        )


# ---------------------------------------------------------------------------
# FusedRMSNorm
# ---------------------------------------------------------------------------

class FusedRMSNorm(torch.nn.Module):
    """Root-Mean-Square Layer Normalisation with optional apex CUDA kernel.

    Implements RMSNorm as defined in Zhang & Sennrich (2019):
    ``y = x / RMS(x) * weight``
    where ``RMS(x) = sqrt(mean(x²) + eps)``.

    Unlike LayerNorm there is no mean subtraction and no bias term, which
    reduces parameter count by one weight vector and is the convention used
    in Llama, Mistral, and most modern open-weight LLMs.

    Three-tier dispatch (highest → lowest priority):
    1. ``apex.normalization.FusedRMSNormAffineFunction`` — single-kernel
       fused forward+backward (apex ≥ 23.08 required).
    2. ``apex.normalization.FusedRMSNormFunction`` — fused without the
       affine weight (weight applied in Python after).
    3. Pure-PyTorch rsqrt fallback — always correct, suitable for A6000.

    Args:
        config: ``TransformerConfig`` containing ``normalization``,
            ``layernorm_zero_centered_gamma``, and ``sequence_parallel``.
        hidden_size: Transformer hidden dimension.
        eps: Epsilon for numerical stability.
        zero_centered_gamma: Initialise weight to 0 (effective gamma = 1).
        normalization: Must be ``"RMSNorm"`` (included to match TE interface).
    """

    def __init__(
        self,
        config: "TransformerConfig",
        hidden_size: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ) -> None:
        super().__init__()

        self.config = config
        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma

        assert self.config.normalization == "RMSNorm", (
            f"FusedRMSNorm only supports 'RMSNorm', got '{self.config.normalization}'"
        )

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps

        # RMSNorm has no bias term.
        self.weight = Parameter(torch.empty(*hidden_size))
        self.reset_parameters()

        self.sequence_parallel = getattr(self.config, "sequence_parallel", False)
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self) -> None:
        """Initialise weight.

        With ``zero_centered_gamma`` the weight is zero-initialised so the
        effective gamma = weight + 1 = 1 at the start of training.
        """
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        """Apply RMS normalisation.

        Dispatches to apex fused kernel or pure-PyTorch fallback.

        Args:
            input: Arbitrary-shape tensor; normalisation is over the last
                ``len(self.hidden_size)`` dimensions.

        Returns:
            Normalised tensor, same shape and dtype as ``input``.
        """
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if HAVE_FUSED_RMS_NORM_AFFINE:
            fwd_args = inspect.getfullargspec(
                FusedRMSNormAffineFunction.forward
            ).args
            if "memory_efficient" in fwd_args:
                return FusedRMSNormAffineFunction.apply(
                    input, weight, self.hidden_size, self.eps,
                    self.config.memory_efficient_layer_norm,
                )
            return FusedRMSNormAffineFunction.apply(
                input, weight, self.hidden_size, self.eps
            )

        if HAVE_FUSED_RMS_NORM:
            # No-affine apex kernel; apply weight in Python.
            fwd_args = inspect.getfullargspec(FusedRMSNormFunction.forward).args
            if "memory_efficient" in fwd_args:
                normed = FusedRMSNormFunction.apply(
                    input, self.hidden_size, self.eps,
                    self.config.memory_efficient_layer_norm,
                )
            else:
                normed = FusedRMSNormFunction.apply(
                    input, self.hidden_size, self.eps
                )
            return normed * weight

        # Neuron_SP hetero CUDA RMSNorm — fast path for BF16 2-D inputs.
        # Uses our fused_swiglu_ln.cu RMSNorm kernel (without SwiGLU — just LN).
        # Requires: BF16 dtype, 2-D [batch, hidden] shape, hidden divisible by 8.
        if (HAVE_HETERO_LN and input.dtype == torch.bfloat16
                and input.dim() == 2
                and input.size(1) % 8 == 0
                and weight.dtype == torch.float32):
            try:
                import torch
                sm_ver = _hetero_ln_sm_version()
                batch, hidden = input.shape
                out = torch.empty_like(input)
                # fused_swiglu_ln computes: out = swiglu(gate, up) * rmsnorm_weight
                # We repurpose fused_layernorm_residual: residual = input, output = normed.
                # Use residual = zeros (no residual add), input = x, weight = rmsnorm weight.
                residual = torch.zeros_like(input)
                _hetero_ln_op.fused_layernorm_residual(
                    out, residual, input, weight, self.eps, sm_ver
                )
                return out
            except Exception:
                pass  # fall through to PyTorch fallback

        # Pure-PyTorch fallback — always correct.
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        input_f32 = input.float()
        variance = input_f32.pow(2).mean(-1, keepdim=True)
        normed = input_f32 * torch.rsqrt(variance + self.eps)
        return (normed.to(input.dtype)) * weight


# ---------------------------------------------------------------------------
# MixedFused convenience wrappers (drop-in torch.nn.LayerNorm replacements).
# ---------------------------------------------------------------------------

class MixedFusedLayerNorm(FusedLayerNorm):
    """Drop-in replacement for ``torch.nn.LayerNorm`` using ``FusedLayerNorm``.

    Accepts ``normalized_shape`` as a positional argument and constructs a
    minimal ``TransformerConfig`` stub so that ``FusedLayerNorm.__init__``
    can be called without requiring callers to build a full config object.

    This mirrors Megatron's ``MixedFusedLayerNorm`` used in legacy model
    code that predates the ``TransformerConfig`` API.

    Args:
        normalized_shape: Integer hidden size or tuple of sizes.
        eps: Epsilon for numerical stability (default 1e-5).
        no_persist_layer_norm: Disable the persistent apex kernel even if
            available (e.g. when benchmarking the fused-but-not-persistent path).
        zero_centered_gamma: Initialise weight to zero (effective gamma = 1).
        normalization: Must be ``"LayerNorm"``; included for interface parity.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        no_persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ) -> None:
        # Build a minimal config stub that satisfies FusedLayerNorm's
        # attribute accesses without a full TransformerConfig import at
        # module level (avoids circular imports in some usage patterns).
        from types import SimpleNamespace
        config = SimpleNamespace(
            normalization="LayerNorm",
            layernorm_zero_centered_gamma=zero_centered_gamma,
            persist_layer_norm=not no_persist_layer_norm,
            memory_efficient_layer_norm=False,
            sequence_parallel=False,
        )
        super().__init__(
            config=config,  # type: ignore[arg-type]
            hidden_size=normalized_shape,
            eps=eps,
            persist_layer_norm=not no_persist_layer_norm,
            zero_centered_gamma=zero_centered_gamma,
            normalization=normalization,
        )


class MixedFusedRMSNorm(FusedRMSNorm):
    """Drop-in replacement for RMSNorm using ``FusedRMSNorm``.

    Accepts ``normalized_shape`` as a positional argument and constructs a
    minimal ``TransformerConfig`` stub matching ``MixedFusedLayerNorm``'s
    pattern, making it a convenient standalone RMSNorm for legacy code.

    Args:
        normalized_shape: Integer hidden size or tuple of sizes.
        eps: Epsilon for numerical stability (default 1e-5).
        zero_centered_gamma: Initialise weight to zero (effective gamma = 1).
        normalization: Must be ``"RMSNorm"``; included for interface parity.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ) -> None:
        from types import SimpleNamespace
        config = SimpleNamespace(
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=zero_centered_gamma,
            memory_efficient_layer_norm=False,
            sequence_parallel=False,
        )
        super().__init__(
            config=config,  # type: ignore[arg-type]
            hidden_size=normalized_shape,
            eps=eps,
            zero_centered_gamma=zero_centered_gamma,
            normalization=normalization,
        )


# ---------------------------------------------------------------------------
# Factory — returns the class (not an instance) for a given config.
# ---------------------------------------------------------------------------

def get_layer_norm(
    config: "TransformerConfig",
) -> type:
    """Return the appropriate fused normalisation *class* for ``config``.

    The caller is responsible for instantiating the returned class with the
    correct ``hidden_size`` and any additional keyword arguments.

    Args:
        config: ``TransformerConfig`` whose ``normalization`` field selects
            the implementation.  Supported values:
            - ``"LayerNorm"`` → ``FusedLayerNorm``
            - ``"RMSNorm"``   → ``FusedRMSNorm``

    Returns:
        The uninstantiated class (``FusedLayerNorm`` or ``FusedRMSNorm``).

    Raises:
        ValueError: If ``config.normalization`` is not a recognised value.

    Example::

        NormClass = get_layer_norm(config)
        norm = NormClass(config, hidden_size=config.hidden_size)
    """
    norm_type = getattr(config, "normalization", "LayerNorm")
    if norm_type == "LayerNorm":
        return FusedLayerNorm
    if norm_type == "RMSNorm":
        return FusedRMSNorm
    raise ValueError(
        f"get_layer_norm: unsupported normalization type '{norm_type}'. "
        f"Expected 'LayerNorm' or 'RMSNorm'."
    )
