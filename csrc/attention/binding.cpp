// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * binding.cpp  —  NeurIPS 2026  DES-LOC + AutoSP  (addresses #135)
 *
 * PyTorch / pybind11 bindings for the fused attention CUDA kernels.
 *
 * Exposed Python API
 * ------------------
 *   ds_fused_attention.fused_attention_forward(
 *       query, key, value,
 *       softmax_scale, causal,
 *       window_left, window_right,
 *       dropout_p, philox_seed, philox_offset,
 *       sm_version
 *   ) -> Tuple[Tensor, Tensor]
 *       query   : BF16 CUDA tensor  [B, Hq, Sq, D]
 *       key     : BF16 CUDA tensor  [B, Hkv, Sk, D]
 *       value   : BF16 CUDA tensor  [B, Hkv, Sk, D]
 *       Returns : (output [B, Hq, Sq, D] BF16, lse [B, Hq, Sq] FP32)
 *
 *   ds_fused_attention.fused_attention_backward(
 *       d_output, query, key, value, output, lse,
 *       softmax_scale, causal, sm_version
 *   ) -> Tuple[Tensor, Tensor, Tensor]
 *       Returns : (dq [B,Hq,Sq,D], dk [B,Hkv,Sk,D], dv [B,Hkv,Sk,D]) — all BF16
 *
 * Layout convention
 * -----------------
 * All tensors use packed row-major layout: [batch, heads, seq, head_dim].
 * This matches the DES-LOC DotProductAttention.forward() convention after the
 * permute/reshape applied in dot_product_attention.py.
 *
 * GQA
 * ---
 * The binding infers gqa_ratio = Hq / Hkv from tensor shapes automatically.
 * No extra argument is needed; the kernel remaps kv_head = q_head / gqa_ratio.
 *
 * SWA (sliding-window attention)
 * -------------------------------
 * Pass window_left >= 0 and/or window_right >= 0 to enable sliding-window
 * masking.  -1 disables the bound in that direction.  causal=True is
 * equivalent to (window_left=-1, window_right=0) and takes precedence.
 *
 * sm_version
 * ----------
 * Pass the integer SM version (e.g. 86, 90, 120).  The binding accepts 0 or
 * -1 as "auto-detect from the current device" — it will call
 * cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor/Minor).
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <tuple>
#include <stdexcept>

#include "fused_attention.h"
#include "fused_gqa_attention.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_bf16_4d(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16,
                name, " must be BFloat16, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(),       name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 4,      name, " must be 4-D [B, H, S, D], got ndim=", t.dim());
}

// Resolve sm_version: 0 or negative → auto-detect from current device.
static int resolve_sm(int sm_version)
{
    if (sm_version > 0) return sm_version;
    int major = 0, minor = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor;
}

// ---------------------------------------------------------------------------
// fused_attention_forward
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor>
fused_attention_forward_py(
    const at::Tensor& query,          // [B, Hq, Sq, D]
    const at::Tensor& key,            // [B, Hkv, Sk, D]
    const at::Tensor& value,          // [B, Hkv, Sk, D]
    float             softmax_scale,
    bool              causal,
    int               window_left,
    int               window_right,
    float             dropout_p,
    int64_t           philox_seed,
    int64_t           philox_offset,
    int               sm_version)
{
    check_bf16_4d(query,  "query");
    check_bf16_4d(key,    "key");
    check_bf16_4d(value,  "value");

    TORCH_CHECK(key.sizes() == value.sizes(),
                "key and value must have the same shape");
    TORCH_CHECK(query.size(0) == key.size(0),
                "query and key must have the same batch size");
    TORCH_CHECK(query.size(3) == key.size(3),
                "query and key must have the same head_dim");

    const int B    = query.size(0);
    const int Hq   = query.size(1);
    const int Sq   = query.size(2);
    const int D    = query.size(3);
    const int Hkv  = key.size(1);
    const int Sk   = key.size(2);

    TORCH_CHECK(Hq % Hkv == 0,
                "num_q_heads (", Hq, ") must be divisible by num_kv_heads (", Hkv, ") for GQA");
    TORCH_CHECK(D % 2 == 0, "head_dim must be divisible by 2, got ", D);

    // Allocate output and LSE
    at::Tensor output = torch::empty_like(query);
    at::Tensor lse    = torch::empty(
        {B, Hq, Sq}, torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int sm = resolve_sm(sm_version);

    launch_fused_attention(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<float*>(lse.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
        B, Hq, Hkv, Sq, Sk, D,
        softmax_scale,
        causal,
        window_left,
        window_right,
        dropout_p,
        static_cast<uint64_t>(philox_seed),
        static_cast<uint64_t>(philox_offset),
        sm,
        stream);

    return {output, lse};
}

// ---------------------------------------------------------------------------
// fused_attention_backward
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_attention_backward_py(
    const at::Tensor& d_output,       // [B, Hq, Sq, D]
    const at::Tensor& query,          // [B, Hq, Sq, D]
    const at::Tensor& key,            // [B, Hkv, Sk, D]
    const at::Tensor& value,          // [B, Hkv, Sk, D]
    const at::Tensor& output,         // [B, Hq, Sq, D]
    const at::Tensor& lse,            // [B, Hq, Sq]  FP32
    float             softmax_scale,
    bool              causal,
    int               sm_version)
{
    check_bf16_4d(d_output, "d_output");
    check_bf16_4d(query,    "query");
    check_bf16_4d(key,      "key");
    check_bf16_4d(value,    "value");
    check_bf16_4d(output,   "output");

    TORCH_CHECK(lse.scalar_type() == at::ScalarType::Float,
                "lse must be Float32, got ", lse.scalar_type());
    TORCH_CHECK(lse.is_cuda(),       "lse must be a CUDA tensor");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
    TORCH_CHECK(lse.dim() == 3,      "lse must be 3-D [B, Hq, Sq], got ndim=", lse.dim());

    const int B   = query.size(0);
    const int Hq  = query.size(1);
    const int Sq  = query.size(2);
    const int D   = query.size(3);
    const int Hkv = key.size(1);
    const int Sk  = key.size(2);

    TORCH_CHECK(Hq % Hkv == 0,
                "num_q_heads (", Hq, ") must be divisible by num_kv_heads (", Hkv, ") for GQA");

    // Allocate gradient tensors (zero-init; kernels accumulate via +=)
    at::Tensor dq = torch::zeros_like(query);
    at::Tensor dk = torch::zeros_like(key);
    at::Tensor dv = torch::zeros_like(value);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int sm = resolve_sm(sm_version);

    launch_fused_attention_bwd(
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
        reinterpret_cast<const float*>(lse.data_ptr()),
        B, Hq, Hkv, Sq, Sk, D,
        softmax_scale,
        causal,
        sm,
        stream);

    return {dq, dk, dv};
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// fused_gqa_attention_forward  (issue #142)
// ---------------------------------------------------------------------------
//
// Exposed Python signature:
//
//   ds_fused_attention.fused_gqa_attention_forward(
//       query, key, value,
//       num_kv_heads,
//       causal=True,
//       sm_scale=0.0,
//       sm_version=0
//   ) -> Tensor
//
// Unlike fused_attention_forward (which dispatches one block per Q-head),
// this entry point dispatches one block per KV-head group and shares the K/V
// tile across all gqa_ratio Q-head warps inside that block.  The result is
// an up-to-gqa_ratio reduction in K/V HBM reads (e.g. 8× for Llama-3-70B).
//
// Returns: output tensor only (no LSE) — GQA forward does not store LSE.
// Use fused_attention_forward if you need LSE for the backward pass.

at::Tensor
fused_gqa_attention_forward_py(
    const at::Tensor& query,      // [B, Hq, Sq, D]  BF16
    const at::Tensor& key,        // [B, Hkv, Sk, D] BF16
    const at::Tensor& value,      // [B, Hkv, Sk, D] BF16
    int               num_kv_heads,
    bool              causal,
    float             sm_scale,
    int               sm_version)
{
    // Input validation
    check_bf16_4d(query,  "query");
    check_bf16_4d(key,    "key");
    check_bf16_4d(value,  "value");

    TORCH_CHECK(key.sizes() == value.sizes(),
                "key and value must have the same shape");
    TORCH_CHECK(query.size(0) == key.size(0),
                "query and key must have the same batch size");
    TORCH_CHECK(query.size(3) == key.size(3),
                "query and key must have the same head_dim");

    const int B    = query.size(0);
    const int Hq   = query.size(1);
    const int Sq   = query.size(2);
    const int D    = query.size(3);
    const int Hkv  = (num_kv_heads > 0) ? num_kv_heads : (int)key.size(1);
    const int Sk   = key.size(2);

    TORCH_CHECK(key.size(1) == Hkv,
                "key num_heads (", key.size(1), ") does not match num_kv_heads (", Hkv, ")");
    TORCH_CHECK(Hq % Hkv == 0,
                "num_q_heads (", Hq, ") must be divisible by num_kv_heads (", Hkv,
                ") for GQA.  gqa_ratio = ", Hq / Hkv);
    TORCH_CHECK(D % 8 == 0,
                "head_dim must be divisible by 8 for float4 vectorised loads, got ", D);

    const int gqa_ratio = Hq / Hkv;
    TORCH_CHECK(gqa_ratio >= 1,
                "gqa_ratio must be >= 1, got ", gqa_ratio);

    // Allocate output
    at::Tensor output = torch::empty_like(query);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int sm = resolve_sm(sm_version);

    launch_fused_gqa_attention(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
        B, Hq, Hkv, Sq, Sk, D,
        sm_scale,
        causal,
        sm,
        stream);

    return output;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() =
        "DES-LOC fused attention CUDA kernels (issues #135, #142).\n"
        "SM-dispatched online-softmax tiled attention for A6000/H100/Blackwell.\n"
        "Includes fused GQA kernel with warp-group KV-sharing (issue #142).";

    m.def("fused_attention_forward",
          &fused_attention_forward_py,
          "Fused BF16 multi-head scaled dot-product attention forward.\n"
          "Returns (output [B,Hq,Sq,D] BF16, lse [B,Hq,Sq] FP32).\n"
          "Supports MHA / GQA / MQA, causal masking, SWA, and dropout.",
          pybind11::arg("query"),
          pybind11::arg("key"),
          pybind11::arg("value"),
          pybind11::arg("softmax_scale"),
          pybind11::arg("causal")         = true,
          pybind11::arg("window_left")    = -1,
          pybind11::arg("window_right")   = -1,
          pybind11::arg("dropout_p")      = 0.0f,
          pybind11::arg("philox_seed")    = 0LL,
          pybind11::arg("philox_offset")  = 0LL,
          pybind11::arg("sm_version")     = 0);

    m.def("fused_attention_backward",
          &fused_attention_backward_py,
          "Fused BF16 multi-head attention backward.\n"
          "Returns (dq, dk, dv) all BF16, accumulated gradients.\n"
          "Caller must pass LSE saved from the forward pass.",
          pybind11::arg("d_output"),
          pybind11::arg("query"),
          pybind11::arg("key"),
          pybind11::arg("value"),
          pybind11::arg("output"),
          pybind11::arg("lse"),
          pybind11::arg("softmax_scale"),
          pybind11::arg("causal")      = true,
          pybind11::arg("sm_version")  = 0);

    m.def("fused_gqa_attention_forward",
          &fused_gqa_attention_forward_py,
          "Fused BF16 Grouped Query Attention (GQA) forward pass.\n"
          "\n"
          "Unlike fused_attention_forward, this kernel dispatches one CUDA block\n"
          "per KV-head group rather than one block per Q-head.  All gqa_ratio\n"
          "Q-head warps within the block share a single K/V tile load, reducing\n"
          "K/V HBM reads by up to gqa_ratio (e.g. 8x for Llama-3-70B).\n"
          "\n"
          "Implementation details (fused_gqa_attention.cu, issue #142):\n"
          "  - Warp-group tiling: warp_id indexes Q-head within the KV group.\n"
          "  - Q tiles in registers (no smem_q); K/V tiles in shared memory.\n"
          "  - Online softmax (Milakov-Gimelshein): single-pass max+sum update.\n"
          "  - Causal bitmask in registers (uint64 over kBc cols): no branches.\n"
          "  - float4 (128-bit) vectorised loads for K/V (8 BF16 per instruction).\n"
          "  - SM dispatch: SM8.6 (__ldg), SM9.0 (cp.async), SM12.0 (cp.async.bulk).\n"
          "  - __launch_bounds__(Policy::kBlockSize, Policy::kMinCTAsPerSM).\n"
          "\n"
          "Args:\n"
          "  query      (Tensor): BF16 CUDA [B, Hq, Sq, D]\n"
          "  key        (Tensor): BF16 CUDA [B, Hkv, Sk, D]\n"
          "  value      (Tensor): BF16 CUDA [B, Hkv, Sk, D]\n"
          "  num_kv_heads (int): Number of KV heads (Hkv). Pass 0 to infer.\n"
          "  causal      (bool): Enable causal masking (default True).\n"
          "  sm_scale   (float): Softmax scale; 0.0 → auto 1/sqrt(D).\n"
          "  sm_version   (int): SM version int (86/90/120); 0 → auto-detect.\n"
          "\n"
          "Returns:\n"
          "  Tensor: BF16 output [B, Hq, Sq, D].",
          pybind11::arg("query"),
          pybind11::arg("key"),
          pybind11::arg("value"),
          pybind11::arg("num_kv_heads") = 0,
          pybind11::arg("causal")       = true,
          pybind11::arg("sm_scale")     = 0.0f,
          pybind11::arg("sm_version")   = 0);
}
