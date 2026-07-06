"""Fused Layer Normalisation.

Mirrors Megatron-LM megatron/core/fusions/fused_layer_norm.py.

Priority of implementations (highest to lowest):
1. ``apex.contrib.layer_norm.FastLayerNormFN`` — persistent CUDA kernel,
   requires ``hidden_size`` in the allow-list *and* apex installed.
2. ``apex.normalization.FusedLayerNormAffineFunction`` — apex fused LN,
   works for any hidden size when apex is present.
3. ``torch.nn.functional.layer_norm`` — pure-PyTorch fallback; always
   available, suitable for SM86 (A6000) on the DES-LOC cluster where apex
   may not be compiled.

The ``FusedLayerNorm`` class accepts a ``TransformerConfig`` (exactly like
Megatron) and is used by ``TESpecProvider.layer_norm`` when TE < 1.9 or for
non-QK layer norms.

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
