# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

######## Fused MoE kernel #########
# These kernels are implemented for
# fusing GeMM with dequantization of
# fp8 weight data when using bit-16
# activation.
###################################

import logging
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TransformerEngine general_gemm wrapper — mirrors Megatron d3528a213
# reinterpreted for DES-LOC heterogeneous clusters:
#
#   SM >= 90 (H100 / GB200)  → FP8 fast path via TE general_gemm
#   SM < 90  (A6000 SM86, A100 SM80, etc.) → BF16/FP16 fallback
#
# The original Megatron fix decoupled workspace from the call-site by
# wrapping get_workspace in a try/except so that newer TE versions that
# removed the kwarg don't crash.  Here we apply the same guard *plus*
# an arch-capability gate so A6000 nodes never attempt FP8 hardware paths
# that cublasLt would reject at SM86.
# ---------------------------------------------------------------------------

try:
    from transformer_engine.pytorch.cpp_extensions import general_gemm as _te_general_gemm

    try:
        from transformer_engine.pytorch.module.base import get_workspace as _get_workspace
    except ImportError:
        # TE >= 1.8 removed get_workspace; workspace is now managed internally.
        _get_workspace = None

    def _sm_major() -> int:
        """Return the SM major version of the current CUDA device, or 0 if CUDA unavailable."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(torch.cuda.current_device()).major

    def te_general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        accumulate: bool = False,
        out: "torch.Tensor | None" = None,
        bias: "torch.Tensor | None" = None,
        use_split_accumulator: bool = False,
    ) -> torch.Tensor:
        """
        Arch-routed wrapper for TE general_gemm.

        Mirrors Megatron d3528a213 (fix TE general_gemm API change) and
        reinterprets the workspace-optionality fix as a per-arch dispatch gate
        for DES-LOC A6000 (SM86) + H100 (SM90) heterogeneous clusters.

        SM >= 90: FP8 hardware path available; call TE general_gemm with the
            optional workspace kwarg guarded by the _get_workspace probe.
        SM < 90 (e.g. SM86 A6000): TE FP8 gemm not supported by cublasLt on
            this arch; fall back to torch.matmul in out_dtype to preserve
            numerical contract.

        Decision point is logged once per unique (device, dtype) pair to give
        visibility into which path each GPU tier actually takes without flooding
        the log on every forward pass.
        """
        sm = _sm_major()

        if sm >= 9:
            # H100 / GB200 / B200: FP8 native path via TE.
            # Workspace kwarg may or may not exist depending on TE version
            # (Megatron d3528a213 taught us to probe rather than assume).
            kwargs = dict(
                out_dtype=out_dtype,
                quantization_params=None,
                gelu=None,
                gelu_input=None,
                grad=accumulate,
                out=out,
                bias=bias,
                use_split_accumulator=use_split_accumulator,
                D_dtype=out_dtype,
                transa=False,
                transb=False,
                ub_algo=None,
                extra_output_buffer=None,
                extra_output=None,
                bulk_overlap=False,
            )
            if _get_workspace is not None:
                kwargs["workspace"] = _get_workspace()

            _log_gemm_route("fp8_te", sm, A, B, out_dtype)
            return _te_general_gemm(A, B, **kwargs)

        else:
            # SM < 90 (A6000 SM86, A100 SM80, V100 SM70, …): cublasLt does not
            # expose FP8 Tensor Core paths below Hopper.  Cast to out_dtype and
            # use torch.matmul which dispatches through cuBLAS TF32 / BF16 on
            # Ampere.
            _log_gemm_route("bf16_fallback", sm, A, B, out_dtype)
            A_cast = A.to(out_dtype)
            B_cast = B.to(out_dtype)
            result = torch.matmul(A_cast, B_cast)
            if bias is not None:
                result = result + bias.to(out_dtype)
            if out is not None:
                if accumulate:
                    out.add_(result)
                else:
                    out.copy_(result)
                return out
            return result

except ImportError:
    te_general_gemm = None  # type: ignore[assignment,misc]
    _get_workspace = None


# ---------------------------------------------------------------------------
# Diagnostic helper — structured, single-emission per (device, path, dtype)
# so the log line is useful for fleet debugging without repeating every step.
# ---------------------------------------------------------------------------

_LOGGED_ROUTES: "set[tuple]" = set()


def _log_gemm_route(path: str, sm: int, A: "torch.Tensor", B: "torch.Tensor",
                    out_dtype: torch.dtype) -> None:
    key = (path, sm, out_dtype, A.device.type)
    if key in _LOGGED_ROUTES:
        return
    _LOGGED_ROUTES.add(key)
    dev_name = (torch.cuda.get_device_properties(torch.cuda.current_device()).name
                if torch.cuda.is_available() else "cpu")
    logger.info(
        "[DES-LOC gemm-route] path=%s sm=%d device=%s out_dtype=%s "
        "A=%s B=%s  — SM≥90 uses TE FP8, SM<90 uses BF16 fallback",
        path, sm, dev_name, out_dtype, tuple(A.shape), tuple(B.shape),
    )


# ---------------------------------------------------------------------------
# Original matmul_fp8 entry points (unchanged contract; arch routing is
# separate from the triton dequant-fused path below)
# ---------------------------------------------------------------------------

def matmul_fp8(inp, weight, scale, quantization_group_size, quantizer):
    from deepspeed import get_accelerator

    if not get_accelerator().is_triton_supported():
        return matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer)
    else:
        # Import dynamically to prevent failures on systems without triton.
        from .fp8_gemm_triton import matmul_fp8_triton
        return matmul_fp8_triton(inp, weight, scale, quantization_group_size)


def matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer):
    return torch.matmul(inp, quantizer.dequantize(weight, scale=scale))
