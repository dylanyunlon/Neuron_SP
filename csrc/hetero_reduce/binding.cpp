// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * binding.cpp
 *
 * PyTorch / pybind11 bindings for the hetero_reduce CUDA kernels.
 *
 * Exposed Python API
 * ------------------
 *   hetero_reduce.fused_bf16_reduce(output, inputs, sm_version) -> None
 *       output : torch.Tensor  BF16, device tensor, shape [N]
 *       inputs : List[torch.Tensor]  BF16 device tensors, each shape [N]
 *       sm_version : int  e.g. 86, 90, 120
 *
 *   hetero_reduce.hetero_reduce_scatter(output, inputs, shard_offset,
 *                                        shard_count, sm_version) -> None
 *       output       : torch.Tensor BF16, device tensor [shard_count]
 *       inputs       : List[torch.Tensor] BF16 device tensors [N]
 *       shard_offset : int  starting element index in the full tensor
 *       shard_count  : int  number of elements this device writes
 *       sm_version   : int  e.g. 86, 90, 120
 *
 *   hetero_reduce.compute_shard_ranges(sm_versions, total_elems)
 *                                        -> List[Tuple[int, int]]
 *       sm_versions : List[int]  per-tier SM versions
 *       total_elems : int        total BF16 elements
 *       Returns list of (offset, count) tuples, one per tier.
 *
 *   hetero_reduce.fused_swiglu_ln(output, gate_proj, up_proj, ln_weight,
 *                                  eps, sm_version) -> None
 *       output     : torch.Tensor  BF16  [batch, hidden]
 *       gate_proj  : torch.Tensor  BF16  [batch, hidden]
 *       up_proj    : torch.Tensor  BF16  [batch, hidden]
 *       ln_weight  : torch.Tensor  FP32  [hidden]
 *       eps        : float
 *       sm_version : int
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

#include "hetero_reduce.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_bf16(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16,
                name, " must be BFloat16, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_fp32(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Float,
                name, " must be Float32, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ---------------------------------------------------------------------------
// fused_bf16_reduce binding
// ---------------------------------------------------------------------------

void fused_bf16_reduce_py(at::Tensor output,
                           std::vector<at::Tensor> inputs,
                           int sm_version)
{
    check_bf16(output, "output");
    TORCH_CHECK(!inputs.empty(), "inputs list must not be empty");
    TORCH_CHECK(inputs.size() <= 32,
                "fused_bf16_reduce supports at most 32 input tensors, got ", inputs.size());

    const size_t n_elems = static_cast<size_t>(output.numel());
    TORCH_CHECK(n_elems % 8 == 0,
                "output numel must be divisible by 8 for vectorised loads, got ", n_elems);

    // Collect raw device pointers.
    std::vector<const __nv_bfloat16*> ptrs;
    ptrs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        check_bf16(inputs[i], ("inputs[" + std::to_string(i) + "]").c_str());
        TORCH_CHECK(static_cast<size_t>(inputs[i].numel()) == n_elems,
                    "inputs[", i, "] numel mismatch: expected ", n_elems,
                    " got ", inputs[i].numel());
        ptrs.push_back(reinterpret_cast<const __nv_bfloat16*>(inputs[i].data_ptr<at::BFloat16>()));
    }

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_bf16_reduce(out_ptr, ptrs.data(),
                              static_cast<int>(ptrs.size()),
                              n_elems, sm_version, stream);
}

// ---------------------------------------------------------------------------
// hetero_reduce_scatter binding
// ---------------------------------------------------------------------------

void hetero_reduce_scatter_py(at::Tensor output,
                               std::vector<at::Tensor> inputs,
                               int64_t shard_offset,
                               int64_t shard_count,
                               int sm_version)
{
    check_bf16(output, "output");
    TORCH_CHECK(!inputs.empty(), "inputs list must not be empty");
    TORCH_CHECK(inputs.size() <= 32,
                "supports at most 32 input tensors, got ", inputs.size());

    TORCH_CHECK(shard_offset >= 0, "shard_offset must be >= 0");
    TORCH_CHECK(shard_count > 0,   "shard_count must be > 0");
    TORCH_CHECK(shard_count % 8 == 0,
                "shard_count must be divisible by 8, got ", shard_count);
    TORCH_CHECK(static_cast<size_t>(output.numel()) >= static_cast<size_t>(shard_count),
                "output numel must be >= shard_count");

    std::vector<const __nv_bfloat16*> ptrs;
    ptrs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        check_bf16(inputs[i], ("inputs[" + std::to_string(i) + "]").c_str());
        TORCH_CHECK(static_cast<size_t>(inputs[i].numel()) >=
                    static_cast<size_t>(shard_offset + shard_count),
                    "inputs[", i, "] numel too small for shard range");
        ptrs.push_back(reinterpret_cast<const __nv_bfloat16*>(inputs[i].data_ptr<at::BFloat16>()));
    }

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_hetero_reduce_scatter(out_ptr, ptrs.data(),
                                  static_cast<int>(ptrs.size()),
                                  static_cast<size_t>(shard_offset),
                                  static_cast<size_t>(shard_count),
                                  sm_version, stream);
}

// ---------------------------------------------------------------------------
// compute_shard_ranges binding
// ---------------------------------------------------------------------------

std::vector<std::tuple<int64_t, int64_t>>
compute_shard_ranges_py(std::vector<int> sm_versions, int64_t total_elems)
{
    TORCH_CHECK(!sm_versions.empty(), "sm_versions must not be empty");
    TORCH_CHECK(total_elems > 0, "total_elems must be > 0");
    TORCH_CHECK(total_elems % 8 == 0,
                "total_elems must be divisible by 8, got ", total_elems);

    const int num_tiers = static_cast<int>(sm_versions.size());
    std::vector<HeteroTierDesc> tiers(num_tiers);
    for (int i = 0; i < num_tiers; i++) {
        tiers[i].device_id   = i;
        tiers[i].sm_version  = sm_versions[i];
        tiers[i].bucket_size = 0;
    }

    std::vector<size_t> offsets(num_tiers);
    std::vector<size_t> counts(num_tiers);
    compute_hetero_shard_ranges(tiers.data(), num_tiers,
                                 static_cast<size_t>(total_elems),
                                 offsets.data(), counts.data());

    std::vector<std::tuple<int64_t, int64_t>> result;
    result.reserve(num_tiers);
    for (int i = 0; i < num_tiers; i++) {
        result.emplace_back(static_cast<int64_t>(offsets[i]),
                            static_cast<int64_t>(counts[i]));
    }
    return result;
}

// ---------------------------------------------------------------------------
// fused_swiglu_ln binding
// ---------------------------------------------------------------------------

void fused_swiglu_ln_py(at::Tensor output,
                         at::Tensor gate_proj,
                         at::Tensor up_proj,
                         at::Tensor ln_weight,
                         float eps,
                         int sm_version)
{
    check_bf16(output,    "output");
    check_bf16(gate_proj, "gate_proj");
    check_bf16(up_proj,   "up_proj");
    check_fp32(ln_weight, "ln_weight");

    TORCH_CHECK(output.dim() == 2,    "output must be 2-D [batch, hidden]");
    TORCH_CHECK(gate_proj.dim() == 2, "gate_proj must be 2-D [batch, hidden]");
    TORCH_CHECK(up_proj.dim() == 2,   "up_proj must be 2-D [batch, hidden]");

    const int batch  = static_cast<int>(output.size(0));
    const int hidden = static_cast<int>(output.size(1));

    TORCH_CHECK(hidden % 8 == 0,
                "hidden must be divisible by 8, got ", hidden);
    TORCH_CHECK(gate_proj.size(0) == batch && gate_proj.size(1) == hidden,
                "gate_proj shape mismatch");
    TORCH_CHECK(up_proj.size(0) == batch && up_proj.size(1) == hidden,
                "up_proj shape mismatch");
    TORCH_CHECK(ln_weight.numel() == hidden,
                "ln_weight must have numel == hidden, got ", ln_weight.numel());

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    const __nv_bfloat16* gate_ptr =
        reinterpret_cast<const __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>());
    const __nv_bfloat16* up_ptr =
        reinterpret_cast<const __nv_bfloat16*>(up_proj.data_ptr<at::BFloat16>());
    const float* w_ptr = ln_weight.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_swiglu_ln(out_ptr, gate_ptr, up_ptr, w_ptr,
                            batch, hidden, eps, sm_version, stream);
}

// ---------------------------------------------------------------------------
// fused_rope_hetero bindings
// ---------------------------------------------------------------------------

void rope_cache_py(at::Tensor cos_cache,
                   at::Tensor sin_cache,
                   int seq_len,
                   int head_dim,
                   float base,
                   int pos_offset)
{
    check_fp32(cos_cache, "cos_cache");
    check_fp32(sin_cache, "sin_cache");
    TORCH_CHECK(cos_cache.is_contiguous(), "cos_cache must be contiguous");
    TORCH_CHECK(sin_cache.is_contiguous(), "sin_cache must be contiguous");
    const int half_dim = head_dim / 2;
    TORCH_CHECK(cos_cache.numel() == (int64_t)seq_len * half_dim,
                "cos_cache size mismatch");
    TORCH_CHECK(sin_cache.numel() == (int64_t)seq_len * half_dim,
                "sin_cache size mismatch");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_rope_cache(cos_cache.data_ptr<float>(),
                      sin_cache.data_ptr<float>(),
                      seq_len, head_dim, base, pos_offset, stream);
}

void fused_rope_hetero_py(at::Tensor output,
                           at::Tensor input,
                           at::Tensor cos_cache,
                           at::Tensor sin_cache,
                           bool neox_style,
                           int sm_version)
{
    check_bf16(output,    "output");
    check_bf16(input,     "input");
    check_fp32(cos_cache, "cos_cache");
    check_fp32(sin_cache, "sin_cache");

    TORCH_CHECK(input.dim() == 4,  "input must be 4-D [B, S, H, D]");
    TORCH_CHECK(output.dim() == 4, "output must be 4-D [B, S, H, D]");
    TORCH_CHECK(output.sizes() == input.sizes(), "output/input shape mismatch");

    const int batch     = (int)input.size(0);
    const int seq_len   = (int)input.size(1);
    const int num_heads = (int)input.size(2);
    const int head_dim  = (int)input.size(3);

    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even, got ", head_dim);

    const int half_dim = head_dim / 2;
    TORCH_CHECK(cos_cache.numel() == (int64_t)seq_len * half_dim,
                "cos_cache numel mismatch");
    TORCH_CHECK(sin_cache.numel() == (int64_t)seq_len * half_dim,
                "sin_cache numel mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_rope_hetero(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        batch, seq_len, num_heads, head_dim,
        neox_style, sm_version, stream);
}

// ---------------------------------------------------------------------------
// pcie_adaptive_allreduce bindings
// ---------------------------------------------------------------------------

/**
 * pcie_gradient_pack_py
 *
 * Gathers non-contiguous gradient slices from multiple BF16 tensors into a
 * flat contiguous BF16 bucket (device-side gather kernel).
 *
 * Each chunk is described by (tensor, byte_offset, length_in_elements).
 * Python-side interface uses a list of (Tensor, int, int) tuples.
 *
 * @param bucket      BF16 output bucket [bucket_elems]
 * @param chunks_in   List of (Tensor, offset_elems, length_elems) tuples
 * @param sm_version  SM version of the active device
 */
void pcie_gradient_pack_py(at::Tensor bucket,
                            std::vector<std::tuple<at::Tensor, int64_t, int64_t>> chunks_in,
                            int sm_version)
{
    check_bf16(bucket, "bucket");
    TORCH_CHECK(!chunks_in.empty(), "chunks must not be empty");

    // Build C-side PcieGradChunk array from Python tuples.
    std::vector<PcieGradChunk> chunks;
    chunks.reserve(chunks_in.size());
    size_t total_elems = 0;
    for (size_t i = 0; i < chunks_in.size(); i++) {
        at::Tensor& t = std::get<0>(chunks_in[i]);
        int64_t offset = std::get<1>(chunks_in[i]);
        int64_t length = std::get<2>(chunks_in[i]);
        check_bf16(t, ("chunks[" + std::to_string(i) + "].tensor").c_str());
        TORCH_CHECK(offset >= 0, "chunk offset must be >= 0");
        TORCH_CHECK(length > 0,  "chunk length must be > 0");
        TORCH_CHECK(length % 8 == 0,
                    "chunk length must be divisible by 8, got ", length);
        TORCH_CHECK(offset + length <= t.numel(),
                    "chunk[", i, "] offset+length exceeds tensor numel");
        PcieGradChunk c;
        c.src    = reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
        c.offset = static_cast<size_t>(offset);
        c.length = static_cast<size_t>(length);
        chunks.push_back(c);
        total_elems += static_cast<size_t>(length);
    }

    TORCH_CHECK(static_cast<size_t>(bucket.numel()) >= total_elems,
                "bucket numel (", bucket.numel(), ") < sum of chunk lengths (", total_elems, ")");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_pcie_gradient_pack(
        reinterpret_cast<__nv_bfloat16*>(bucket.data_ptr<at::BFloat16>()),
        chunks.data(),
        static_cast<int>(chunks.size()),
        total_elems,
        sm_version,
        stream);
}

at::Tensor pcie_ring_reduce_py(at::Tensor dst,
                                at::Tensor src,
                                int sm_version)
{
    check_bf16(dst, "dst");
    check_bf16(src, "src");
    TORCH_CHECK(dst.numel() == src.numel(), "dst/src numel mismatch");
    TORCH_CHECK(dst.numel() % 8 == 0, "numel must be divisible by 8");
    TORCH_CHECK(dst.numel() > 0, "dst must be non-empty");
    TORCH_CHECK(dst.device() == src.device(),
                "dst and src must be on the same CUDA device");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_pcie_ring_reduce(
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        (size_t)dst.numel(), sm_version, stream);
    return dst;
}

void pcie_allreduce_finalise_py(at::Tensor out,
                                 at::Tensor src,
                                 int world_size,
                                 int sm_version)
{
    check_bf16(out, "out");
    check_bf16(src, "src");
    TORCH_CHECK(out.numel() == src.numel(), "out/src numel mismatch");
    TORCH_CHECK(out.numel() % 8 == 0, "numel must be divisible by 8");
    TORCH_CHECK(out.numel() > 0, "out must be non-empty");
    TORCH_CHECK(out.device() == src.device(),
                "out and src must be on the same CUDA device");
    TORCH_CHECK(world_size > 0, "world_size must be > 0");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_pcie_allreduce_finalise(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        (size_t)out.numel(), world_size, sm_version, stream);
}

int64_t pcie_bucket_size_py(float pcie_bw_gbps)
{
    return (int64_t)compute_pcie_bucket_size(pcie_bw_gbps);
}

// ---------------------------------------------------------------------------
// tier_activation_offload bindings
// ---------------------------------------------------------------------------

void activation_pack_py(at::Tensor output,
                         std::vector<at::Tensor> inputs,
                         int sm_version)
{
    check_bf16(output, "output");
    TORCH_CHECK(!inputs.empty(), "inputs must not be empty");
    const size_t tensor_elems = (size_t)inputs[0].numel();
    TORCH_CHECK(tensor_elems % 8 == 0, "tensor_elems must be divisible by 8");
    TORCH_CHECK((size_t)output.numel() == inputs.size() * tensor_elems,
                "output size mismatch");

    std::vector<const __nv_bfloat16*> ptrs;
    ptrs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        check_bf16(inputs[i], ("inputs[" + std::to_string(i) + "]").c_str());
        TORCH_CHECK((size_t)inputs[i].numel() == tensor_elems,
                    "inputs[", i, "] numel mismatch");
        ptrs.push_back(reinterpret_cast<const __nv_bfloat16*>(
            inputs[i].data_ptr<at::BFloat16>()));
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_activation_pack(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        ptrs.data(), (int)ptrs.size(), tensor_elems, sm_version, stream);
}

void activation_unpack_py(std::vector<at::Tensor> outputs,
                           at::Tensor flat,
                           int sm_version)
{
    check_bf16(flat, "flat");
    TORCH_CHECK(!outputs.empty(), "outputs must not be empty");
    const size_t tensor_elems = (size_t)outputs[0].numel();
    TORCH_CHECK(tensor_elems % 8 == 0, "tensor_elems must be divisible by 8");
    TORCH_CHECK((size_t)flat.numel() == outputs.size() * tensor_elems,
                "flat size mismatch");

    std::vector<__nv_bfloat16*> ptrs;
    ptrs.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
        check_bf16(outputs[i], ("outputs[" + std::to_string(i) + "]").c_str());
        TORCH_CHECK((size_t)outputs[i].numel() == tensor_elems,
                    "outputs[", i, "] numel mismatch");
        ptrs.push_back(reinterpret_cast<__nv_bfloat16*>(
            outputs[i].data_ptr<at::BFloat16>()));
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_activation_unpack(
        ptrs.data(),
        reinterpret_cast<const __nv_bfloat16*>(flat.data_ptr<at::BFloat16>()),
        (int)ptrs.size(), tensor_elems, sm_version, stream);
}

void quantise_bf16_to_int8_py(at::Tensor output,
                                at::Tensor scales,
                                at::Tensor input)
{
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Char,
                "output must be Int8");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous());
    check_fp32(scales, "scales");
    check_bf16(input,  "input");
    TORCH_CHECK(output.numel() == input.numel(), "output/input numel mismatch");

    const size_t n_elems = (size_t)input.numel();
    const size_t n_tiles = (n_elems + 127) / 128;
    TORCH_CHECK((size_t)scales.numel() >= n_tiles,
                "scales buffer too small: need ", n_tiles, " got ", scales.numel());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_quantise_fp16_to_int8(
        reinterpret_cast<int8_t*>(output.data_ptr<int8_t>()),
        scales.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        n_elems, stream);
}

void dequantise_int8_to_bf16_py(at::Tensor output,
                                  at::Tensor input,
                                  at::Tensor scales)
{
    check_bf16(output, "output");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Char,
                "input must be Int8");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    check_fp32(scales, "scales");
    TORCH_CHECK(output.numel() == input.numel(), "output/input numel mismatch");

    const size_t n_elems = (size_t)input.numel();
    const size_t n_tiles = (n_elems + 127) / 128;
    TORCH_CHECK((size_t)scales.numel() >= n_tiles,
                "scales buffer too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_dequantise_int8_to_fp16(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const int8_t*>(input.data_ptr<int8_t>()),
        scales.data_ptr<float>(),
        n_elems, stream);
}

int64_t compute_offload_budget_py(int64_t total_act_bytes,
                                   int64_t vram_free_bytes,
                                   float   headroom_frac)
{
    return (int64_t)compute_offload_budget(
        (size_t)total_act_bytes, (size_t)vram_free_bytes, headroom_frac);
}

// ---------------------------------------------------------------------------
// fused_layernorm_residual binding  (#110)
// ---------------------------------------------------------------------------

void fused_layernorm_residual_py(at::Tensor output,
                                   at::Tensor residual,
                                   at::Tensor input,
                                   at::Tensor ln_weight,
                                   float      eps,
                                   int        sm_version)
{
    check_bf16(output,   "output");
    check_bf16(residual, "residual");
    check_bf16(input,    "input");
    check_fp32(ln_weight, "ln_weight");

    TORCH_CHECK(output.dim() == 2,   "output must be 2-D [batch, hidden]");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2-D [batch, hidden]");
    TORCH_CHECK(input.dim() == 2,    "input must be 2-D [batch, hidden]");

    const int batch  = (int)output.size(0);
    const int hidden = (int)output.size(1);

    TORCH_CHECK(hidden % 8 == 0,
                "hidden must be divisible by 8, got ", hidden);
    TORCH_CHECK(residual.sizes() == output.sizes(), "residual/output shape mismatch");
    TORCH_CHECK(input.sizes()    == output.sizes(), "input/output shape mismatch");
    TORCH_CHECK(ln_weight.numel() == hidden,
                "ln_weight numel must equal hidden, got ", ln_weight.numel());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_layernorm_residual(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        ln_weight.data_ptr<float>(),
        batch, hidden, eps, sm_version, stream);
}

// ---------------------------------------------------------------------------
// cross_entropy_tp bindings  (#110)
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cross_entropy_tp_forward_py(at::Tensor logits,
                              at::Tensor labels,
                              int64_t    shard_offset,
                              int        sm_version)
{
    check_bf16(logits, "logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, v_local]");
    TORCH_CHECK(labels.scalar_type() == at::ScalarType::Int,
                "labels must be Int32, got ", labels.scalar_type());
    TORCH_CHECK(labels.is_cuda() && labels.is_contiguous(),
                "labels must be a contiguous CUDA tensor");
    TORCH_CHECK(labels.numel() == logits.size(0),
                "labels numel must equal batch size");

    const int batch   = (int)logits.size(0);
    const int v_local = (int)logits.size(1);

    TORCH_CHECK(shard_offset >= 0, "shard_offset must be >= 0");

    auto opts = at::TensorOptions().dtype(at::kFloat).device(logits.device());
    at::Tensor local_max     = at::empty({batch}, opts);
    at::Tensor local_sum_exp = at::empty({batch}, opts);
    at::Tensor local_logit   = at::zeros({batch}, opts);  // zeros for non-shard labels

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cross_entropy_tp_forward(
        local_max.data_ptr<float>(),
        local_sum_exp.data_ptr<float>(),
        local_logit.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        labels.data_ptr<int>(),
        batch, v_local, (int)shard_offset, sm_version, stream);

    return std::make_tuple(local_max, local_sum_exp, local_logit);
}

at::Tensor cross_entropy_tp_loss_py(at::Tensor global_max,
                                     at::Tensor global_sum_exp,
                                     at::Tensor global_logit)
{
    check_fp32(global_max,     "global_max");
    check_fp32(global_sum_exp, "global_sum_exp");
    check_fp32(global_logit,   "global_logit");
    TORCH_CHECK(global_max.numel()     == global_sum_exp.numel(), "shape mismatch");
    TORCH_CHECK(global_max.numel()     == global_logit.numel(),   "shape mismatch");

    const int batch = (int)global_max.numel();
    at::Tensor loss = at::empty({batch},
        at::TensorOptions().dtype(at::kFloat).device(global_max.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cross_entropy_tp_loss(
        loss.data_ptr<float>(),
        global_max.data_ptr<float>(),
        global_sum_exp.data_ptr<float>(),
        global_logit.data_ptr<float>(),
        batch, stream);
    return loss;
}

void cross_entropy_tp_backward_py(at::Tensor d_logits,
                                    at::Tensor logits,
                                    at::Tensor labels,
                                    at::Tensor global_max,
                                    at::Tensor log_sum_exp,
                                    int64_t    shard_offset,
                                    float      inv_batch,
                                    int        sm_version)
{
    check_bf16(d_logits,   "d_logits");
    check_bf16(logits,     "logits");
    check_fp32(global_max, "global_max");
    check_fp32(log_sum_exp, "log_sum_exp");
    TORCH_CHECK(labels.scalar_type() == at::ScalarType::Int,
                "labels must be Int32");
    TORCH_CHECK(d_logits.sizes() == logits.sizes(), "d_logits/logits shape mismatch");

    const int batch   = (int)logits.size(0);
    const int v_local = (int)logits.size(1);
    TORCH_CHECK(shard_offset >= 0, "shard_offset must be >= 0");
    TORCH_CHECK(inv_batch > 0.f,   "inv_batch must be > 0");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cross_entropy_tp_backward(
        reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        labels.data_ptr<int>(),
        global_max.data_ptr<float>(),
        log_sum_exp.data_ptr<float>(),
        batch, v_local, (int)shard_offset, inv_batch, sm_version, stream);
}

// ---------------------------------------------------------------------------
// cross_entropy_tp heterogeneous vocab partition bindings  (#141)
// ---------------------------------------------------------------------------

// compute_hetero_vocab_partition_py
//   sm_versions: list[int] of length tp_size
//   vocab_size:  int
//   returns:     list of dicts {v_local, shard_offset, tp_size, rank}
py::list compute_hetero_vocab_partition_py(
    std::vector<int> sm_versions,
    int              vocab_size)
{
    const int tp_size = (int)sm_versions.size();
    TORCH_CHECK(tp_size >= 1, "sm_versions must be non-empty");
    TORCH_CHECK(vocab_size >= tp_size,
                "vocab_size must be >= tp_size");

    std::vector<VocabPartition> parts(tp_size);
    compute_hetero_vocab_partition(parts.data(), sm_versions.data(),
                                    tp_size, vocab_size);

    py::list result;
    for (int r = 0; r < tp_size; ++r) {
        py::dict d;
        d["v_local"]      = parts[r].v_local;
        d["shard_offset"] = parts[r].shard_offset;
        d["tp_size"]      = parts[r].tp_size;
        d["rank"]         = parts[r].rank;
        result.append(d);
    }
    return result;
}

// cross_entropy_tp_forward_hetero_py
//   logits:       BF16 Tensor [batch, v_local]  — this rank's shard
//   labels:       Int32 Tensor [batch]           — global vocab label
//   v_local:      int   — number of vocab tokens on this rank
//   shard_offset: int   — first global vocab index on this rank
//   sm_version:   int
//   returns:      Tuple[Tensor FP32 [B], Tensor FP32 [B], Tensor FP32 [B]]
std::tuple<at::Tensor, at::Tensor, at::Tensor>
cross_entropy_tp_forward_hetero_py(
    at::Tensor logits,
    at::Tensor labels,
    int64_t    v_local,
    int64_t    shard_offset,
    int        sm_version)
{
    check_bf16(logits, "logits");
    TORCH_CHECK(logits.dim() == 2,
                "logits must be 2-D [batch, v_local]");
    TORCH_CHECK(labels.scalar_type() == at::ScalarType::Int,
                "labels must be Int32, got ", labels.scalar_type());
    TORCH_CHECK(labels.is_cuda() && labels.is_contiguous(),
                "labels must be a contiguous CUDA tensor");
    TORCH_CHECK(labels.numel() == logits.size(0),
                "labels numel must equal batch size");
    TORCH_CHECK(v_local > 0,        "v_local must be > 0");
    TORCH_CHECK(shard_offset >= 0,  "shard_offset must be >= 0");
    TORCH_CHECK((int64_t)logits.size(1) == v_local,
                "logits.size(1) must equal v_local");

    const int batch = (int)logits.size(0);

    auto opts = at::TensorOptions().dtype(at::kFloat).device(logits.device());
    at::Tensor local_max     = at::empty({batch}, opts);
    at::Tensor local_sum_exp = at::empty({batch}, opts);
    at::Tensor local_logit   = at::zeros({batch}, opts);

    VocabPartition vp;
    vp.v_local      = (int)v_local;
    vp.shard_offset = (int)shard_offset;
    vp.tp_size      = 0;   // not needed by the kernel
    vp.rank         = 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cross_entropy_tp_forward_hetero(
        local_max.data_ptr<float>(),
        local_sum_exp.data_ptr<float>(),
        local_logit.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        labels.data_ptr<int>(),
        batch, vp, sm_version, stream);

    return std::make_tuple(local_max, local_sum_exp, local_logit);
}

// cross_entropy_tp_backward_hetero_py
void cross_entropy_tp_backward_hetero_py(
    at::Tensor d_logits,
    at::Tensor logits,
    at::Tensor labels,
    at::Tensor global_max,
    at::Tensor log_sum_exp,
    int64_t    v_local,
    int64_t    shard_offset,
    float      inv_batch,
    int        sm_version)
{
    check_bf16(d_logits,    "d_logits");
    check_bf16(logits,      "logits");
    check_fp32(global_max,  "global_max");
    check_fp32(log_sum_exp, "log_sum_exp");
    TORCH_CHECK(labels.scalar_type() == at::ScalarType::Int,
                "labels must be Int32");
    TORCH_CHECK(d_logits.sizes() == logits.sizes(),
                "d_logits/logits shape mismatch");
    TORCH_CHECK(v_local > 0,        "v_local must be > 0");
    TORCH_CHECK(shard_offset >= 0,  "shard_offset must be >= 0");
    TORCH_CHECK((int64_t)logits.size(1) == v_local,
                "logits.size(1) must equal v_local");
    TORCH_CHECK(inv_batch > 0.f, "inv_batch must be > 0");

    const int batch = (int)logits.size(0);

    VocabPartition vp;
    vp.v_local      = (int)v_local;
    vp.shard_offset = (int)shard_offset;
    vp.tp_size      = 0;
    vp.rank         = 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cross_entropy_tp_backward_hetero(
        reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        labels.data_ptr<int>(),
        global_max.data_ptr<float>(),
        log_sum_exp.data_ptr<float>(),
        batch, vp, inv_batch, sm_version, stream);
}

// ---------------------------------------------------------------------------
// fused_adam_heterogeneous bindings
// ---------------------------------------------------------------------------

void fused_adam_heterogeneous_py(
    at::Tensor  params,
    at::Tensor  exp_avg,
    at::Tensor  exp_avg_sq,
    at::Tensor  grads,
    float       lr_base,
    float       lr_scale,
    float       beta1,
    float       beta2,
    float       bc1,
    float       bc2,
    float       eps,
    float       weight_decay,
    int         sm_version,
    at::Tensor  master_params_opt)   // pass empty tensor to disable
{
    check_bf16(params,  "params");
    check_bf16(grads,   "grads");
    check_fp32(exp_avg,    "exp_avg");
    check_fp32(exp_avg_sq, "exp_avg_sq");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(),    "exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "exp_avg_sq must be contiguous");
    TORCH_CHECK(grads.is_contiguous(),      "grads must be contiguous");

    const size_t n_elems = static_cast<size_t>(params.numel());
    TORCH_CHECK(static_cast<size_t>(exp_avg.numel())    == n_elems, "exp_avg numel mismatch");
    TORCH_CHECK(static_cast<size_t>(exp_avg_sq.numel()) == n_elems, "exp_avg_sq numel mismatch");
    TORCH_CHECK(static_cast<size_t>(grads.numel())      == n_elems, "grads numel mismatch");
    TORCH_CHECK(lr_base  > 0.f, "lr_base must be positive");
    TORCH_CHECK(lr_scale > 0.f, "lr_scale must be positive");
    TORCH_CHECK(bc1 > 0.f, "bc1 must be positive (step > 0?)");
    TORCH_CHECK(bc2 > 0.f, "bc2 must be positive (step > 0?)");
    TORCH_CHECK(eps > 0.f, "eps must be positive");

    float* master_ptr = nullptr;
    if (master_params_opt.defined() && master_params_opt.numel() > 0) {
        check_fp32(master_params_opt, "master_params");
        TORCH_CHECK(master_params_opt.is_contiguous(), "master_params must be contiguous");
        TORCH_CHECK(static_cast<size_t>(master_params_opt.numel()) == n_elems,
                    "master_params numel mismatch");
        master_ptr = master_params_opt.data_ptr<float>();
    }

    __nv_bfloat16* params_ptr =
        reinterpret_cast<__nv_bfloat16*>(params.data_ptr<at::BFloat16>());
    const __nv_bfloat16* grads_ptr =
        reinterpret_cast<const __nv_bfloat16*>(grads.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_adam_heterogeneous(
        params_ptr, master_ptr,
        exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
        grads_ptr, n_elems,
        lr_base, lr_scale,
        beta1, beta2, bc1, bc2, eps, weight_decay,
        sm_version, stream);
}

float hetero_adam_lr_scale_py(int sm_version)
{
    return hetero_adam_lr_scale(sm_version);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// hetero_bucket_size_elems binding
// ---------------------------------------------------------------------------

int64_t hetero_bucket_size_elems_py(int sm_version)
{
    return (int64_t)hetero_bucket_size_elems(sm_version);
}

// ---------------------------------------------------------------------------
// compute_adaptive_chunk_size binding
// ---------------------------------------------------------------------------

int64_t compute_adaptive_chunk_size_py(float pcie_bw_gbps)
{
    return (int64_t)compute_adaptive_chunk_size(pcie_bw_gbps);
}

// ---------------------------------------------------------------------------
// probe_pcie_bandwidth binding
// ---------------------------------------------------------------------------

float probe_pcie_bandwidth_py(int src_device, int dst_device)
{
    return probe_pcie_bandwidth(src_device, dst_device);
}

// ---------------------------------------------------------------------------
// pcie_ring_reduce_step binding (single ring-allreduce step, double-buffer)
// ---------------------------------------------------------------------------

void pcie_ring_reduce_step_py(at::Tensor accum_buf,
                               at::Tensor recv_buf,
                               int sm_version)
{
    check_bf16(accum_buf, "accum_buf");
    check_bf16(recv_buf,  "recv_buf");
    TORCH_CHECK(accum_buf.numel() == recv_buf.numel(),
                "accum_buf/recv_buf numel mismatch");
    TORCH_CHECK(accum_buf.numel() % 8 == 0,
                "numel must be divisible by 8, got ", accum_buf.numel());
    TORCH_CHECK(accum_buf.numel() > 0, "accum_buf must be non-empty");
    TORCH_CHECK(accum_buf.device() == recv_buf.device(),
                "accum_buf and recv_buf must be on the same CUDA device");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_pcie_ring_reduce_step(
        reinterpret_cast<__nv_bfloat16*>(accum_buf.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(recv_buf.data_ptr<at::BFloat16>()),
        (size_t)accum_buf.numel(), sm_version, stream);
}


// ---------------------------------------------------------------------------
// launch_fused_layernorm_residual_ex binding
// ---------------------------------------------------------------------------

void fused_layernorm_residual_ex_py(
    at::Tensor output,
    at::Tensor residual,
    at::Tensor input,
    at::Tensor ln_weight,
    float      eps,
    bool       full_ln,
    int        sm_version,
    py::object bias_obj,
    py::object output_fp32_obj)
{
    check_bf16(output,   "output");
    check_bf16(residual, "residual");
    check_bf16(input,    "input");
    check_fp32(ln_weight, "ln_weight");
    TORCH_CHECK(output.dim() == 2, "output must be 2-D");
    const int batch  = (int)output.size(0);
    const int hidden = (int)output.size(1);
    TORCH_CHECK(hidden % 8 == 0, "hidden must be divisible by 8");
    TORCH_CHECK(residual.sizes() == output.sizes(), "residual/output shape mismatch");
    TORCH_CHECK(input.sizes()    == output.sizes(), "input/output shape mismatch");

    // Optional bias (BF16 [hidden] or None).
    const __nv_bfloat16* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        at::Tensor bias = bias_obj.cast<at::Tensor>();
        check_bf16(bias, "bias");
        TORCH_CHECK(bias.numel() == hidden, "bias numel must equal hidden");
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>());
    }

    // Optional FP32 output buffer.
    float* fp32_ptr = nullptr;
    if (!output_fp32_obj.is_none()) {
        at::Tensor out32 = output_fp32_obj.cast<at::Tensor>();
        check_fp32(out32, "output_fp32");
        TORCH_CHECK(out32.sizes() == output.sizes(), "output_fp32/output shape mismatch");
        fp32_ptr = out32.data_ptr<float>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_layernorm_residual_ex(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        bias_ptr, ln_weight.data_ptr<float>(), fp32_ptr,
        batch, hidden, eps, full_ln, sm_version, stream);
}

// ---------------------------------------------------------------------------
// launch_fused_rope_cacheless binding
// ---------------------------------------------------------------------------

void fused_rope_cacheless_py(
    at::Tensor output,
    at::Tensor input,
    float      base,
    int        pos_offset,
    bool       neox_style,
    int        sm_version)
{
    check_bf16(output, "output");
    check_bf16(input,  "input");
    TORCH_CHECK(input.dim() == 4,  "input must be 4-D [B, S, H, D]");
    TORCH_CHECK(output.dim() == 4, "output must be 4-D [B, S, H, D]");
    TORCH_CHECK(output.sizes() == input.sizes(), "output/input shape mismatch");
    const int batch     = (int)input.size(0);
    const int seq_len   = (int)input.size(1);
    const int num_heads = (int)input.size(2);
    const int head_dim  = (int)input.size(3);
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_rope_cacheless(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        batch, seq_len, num_heads, head_dim,
        base, pos_offset, neox_style, sm_version, stream);
}

// ---------------------------------------------------------------------------
// launch_grad_norm_sq_fp8 binding
// ---------------------------------------------------------------------------

void grad_norm_sq_fp8_py(
    at::Tensor    grads,
    at::Tensor    norm_sq_accum,
    float         fp8_scale,
    int           sm_version)
{
    TORCH_CHECK(grads.scalar_type() == at::ScalarType::Char ||
                grads.scalar_type() == at::ScalarType::Byte,
                "grads must be Int8 or Byte for FP8");
    TORCH_CHECK(grads.is_cuda() && grads.is_contiguous());
    check_fp32(norm_sq_accum, "norm_sq_accum");
    TORCH_CHECK(norm_sq_accum.numel() == 1, "norm_sq_accum must be scalar");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_grad_norm_sq_fp8(
        reinterpret_cast<const uint8_t*>(grads.data_ptr()),
        (size_t)grads.numel(),
        norm_sq_accum.data_ptr<float>(),
        fp8_scale, sm_version, stream);
}

// ---------------------------------------------------------------------------
// fused_gradient_allreduce bindings  (#146)
// ---------------------------------------------------------------------------

void gradient_compress_py(at::Tensor output_int8,
                           at::Tensor output_scale,
                           at::Tensor input,
                           int sm_version)
{
    TORCH_CHECK(output_int8.scalar_type() == at::ScalarType::Char,
                "output_int8 must be Int8");
    TORCH_CHECK(output_int8.is_cuda() && output_int8.is_contiguous());
    check_fp32(output_scale, "output_scale");
    check_bf16(input, "input");

    const size_t n_elems = (size_t)input.numel();
    TORCH_CHECK((size_t)output_int8.numel() >= n_elems,
                "output_int8 too small");
    const size_t n_blocks = (n_elems + 255) / 256;
    TORCH_CHECK((size_t)output_scale.numel() >= n_blocks,
                "output_scale too small: need ", n_blocks, " got ", output_scale.numel());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gradient_compress(
        reinterpret_cast<int8_t*>(output_int8.data_ptr<int8_t>()),
        output_scale.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        n_elems, sm_version, stream);
}

void int8_ring_reduce_step_py(at::Tensor dst_int8,
                                at::Tensor dst_scale,
                                at::Tensor src_int8,
                                at::Tensor src_scale,
                                int sm_version)
{
    TORCH_CHECK(dst_int8.scalar_type() == at::ScalarType::Char, "dst_int8 must be Int8");
    TORCH_CHECK(src_int8.scalar_type() == at::ScalarType::Char, "src_int8 must be Int8");
    TORCH_CHECK(dst_int8.is_cuda() && dst_int8.is_contiguous());
    TORCH_CHECK(src_int8.is_cuda() && src_int8.is_contiguous());
    check_fp32(dst_scale, "dst_scale");
    check_fp32(src_scale, "src_scale");
    TORCH_CHECK(dst_int8.numel() == src_int8.numel(), "dst/src int8 numel mismatch");
    TORCH_CHECK(dst_scale.numel() == src_scale.numel(), "dst/src scale numel mismatch");

    const size_t n_elems = (size_t)dst_int8.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_int8_ring_reduce_step(
        reinterpret_cast<int8_t*>(dst_int8.data_ptr<int8_t>()),
        dst_scale.data_ptr<float>(),
        reinterpret_cast<const int8_t*>(src_int8.data_ptr<int8_t>()),
        src_scale.data_ptr<float>(),
        n_elems, sm_version, stream);
}

void gradient_decompress_py(at::Tensor output,
                              at::Tensor int8_data,
                              at::Tensor scale_buf,
                              int sm_version)
{
    check_bf16(output, "output");
    TORCH_CHECK(int8_data.scalar_type() == at::ScalarType::Char, "int8_data must be Int8");
    TORCH_CHECK(int8_data.is_cuda() && int8_data.is_contiguous());
    check_fp32(scale_buf, "scale_buf");
    TORCH_CHECK(output.numel() == int8_data.numel(), "output/int8_data numel mismatch");

    const size_t n_elems = (size_t)output.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gradient_decompress(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const int8_t*>(int8_data.data_ptr<int8_t>()),
        scale_buf.data_ptr<float>(),
        n_elems, sm_version, stream);
}

void gradient_allreduce_finalise_py(at::Tensor scale_buf,
                                      int64_t n_elems,
                                      int world_size)
{
    check_fp32(scale_buf, "scale_buf");
    TORCH_CHECK(n_elems > 0, "n_elems must be > 0");
    TORCH_CHECK(world_size > 0, "world_size must be > 0");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gradient_allreduce_finalise(
        scale_buf.data_ptr<float>(),
        (size_t)n_elems, world_size, stream);
}

int64_t gradient_compress_bytes_py(int64_t n_elems)
{
    return (int64_t)gradient_compress_bytes((size_t)n_elems);
}

int64_t gradient_scale_bytes_py(int64_t n_elems)
{
    return (int64_t)gradient_scale_bytes((size_t)n_elems);
}

// ---------------------------------------------------------------------------
// fused_swiglu_ln forward-save & backward bindings  (#146)
// ---------------------------------------------------------------------------

void fused_swiglu_ln_fwd_save_py(at::Tensor output,
                                   at::Tensor rms_inv_out,
                                   at::Tensor gate_proj,
                                   at::Tensor up_proj,
                                   at::Tensor ln_weight,
                                   float eps,
                                   int sm_version)
{
    check_bf16(output,    "output");
    check_fp32(rms_inv_out, "rms_inv_out");
    check_bf16(gate_proj, "gate_proj");
    check_bf16(up_proj,   "up_proj");
    check_fp32(ln_weight, "ln_weight");

    TORCH_CHECK(output.dim() == 2,    "output must be 2-D [batch, hidden]");
    TORCH_CHECK(gate_proj.dim() == 2, "gate_proj must be 2-D");
    TORCH_CHECK(up_proj.dim() == 2,   "up_proj must be 2-D");

    const int batch  = (int)output.size(0);
    const int hidden = (int)output.size(1);
    TORCH_CHECK(hidden % 8 == 0, "hidden must be divisible by 8");
    TORCH_CHECK(gate_proj.size(0) == batch && gate_proj.size(1) == hidden, "gate_proj shape mismatch");
    TORCH_CHECK(up_proj.size(0) == batch && up_proj.size(1) == hidden, "up_proj shape mismatch");
    TORCH_CHECK(ln_weight.numel() == hidden, "ln_weight numel must equal hidden");
    TORCH_CHECK(rms_inv_out.numel() == batch, "rms_inv_out must have numel == batch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_swiglu_ln_fwd_save(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        rms_inv_out.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_proj.data_ptr<at::BFloat16>()),
        ln_weight.data_ptr<float>(),
        batch, hidden, eps, sm_version, stream);
}

void fused_swiglu_ln_backward_py(at::Tensor d_gate,
                                   at::Tensor d_up,
                                   at::Tensor d_ln_weight,
                                   at::Tensor d_output,
                                   at::Tensor gate_proj,
                                   at::Tensor up_proj,
                                   at::Tensor ln_weight,
                                   at::Tensor rms_inv_buf,
                                   float eps,
                                   int sm_version)
{
    check_bf16(d_gate,     "d_gate");
    check_bf16(d_up,       "d_up");
    check_fp32(d_ln_weight, "d_ln_weight");
    check_bf16(d_output,   "d_output");
    check_bf16(gate_proj,  "gate_proj");
    check_bf16(up_proj,    "up_proj");
    check_fp32(ln_weight,  "ln_weight");
    check_fp32(rms_inv_buf, "rms_inv_buf");

    TORCH_CHECK(d_output.dim() == 2, "d_output must be 2-D [batch, hidden]");
    const int batch  = (int)d_output.size(0);
    const int hidden = (int)d_output.size(1);
    TORCH_CHECK(hidden % 8 == 0, "hidden must be divisible by 8");
    TORCH_CHECK(d_gate.sizes() == d_output.sizes(), "d_gate shape mismatch");
    TORCH_CHECK(d_up.sizes() == d_output.sizes(), "d_up shape mismatch");
    TORCH_CHECK(gate_proj.sizes() == d_output.sizes(), "gate_proj shape mismatch");
    TORCH_CHECK(up_proj.sizes() == d_output.sizes(), "up_proj shape mismatch");
    TORCH_CHECK(d_ln_weight.numel() == hidden, "d_ln_weight numel must equal hidden");
    TORCH_CHECK(ln_weight.numel() == hidden, "ln_weight numel must equal hidden");
    TORCH_CHECK(rms_inv_buf.numel() == batch, "rms_inv_buf numel must equal batch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_swiglu_ln_backward(
        reinterpret_cast<__nv_bfloat16*>(d_gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(d_up.data_ptr<at::BFloat16>()),
        d_ln_weight.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_proj.data_ptr<at::BFloat16>()),
        ln_weight.data_ptr<float>(),
        rms_inv_buf.data_ptr<float>(),
        batch, hidden, eps, sm_version, stream);
}

// ---------------------------------------------------------------------------
// grad_norm_sq binding  (#146)
// ---------------------------------------------------------------------------

void grad_norm_sq_py(at::Tensor grads,
                      at::Tensor norm_sq_accum,
                      int sm_version)
{
    check_bf16(grads, "grads");
    check_fp32(norm_sq_accum, "norm_sq_accum");
    TORCH_CHECK(norm_sq_accum.numel() == 1, "norm_sq_accum must be scalar");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_grad_norm_sq(
        reinterpret_cast<const __nv_bfloat16*>(grads.data_ptr<at::BFloat16>()),
        (size_t)grads.numel(),
        norm_sq_accum.data_ptr<float>(),
        sm_version, stream);
}

// ---------------------------------------------------------------------------
// fused_adamw_amsgrad_heterogeneous binding  (#146)
// ---------------------------------------------------------------------------

void fused_adamw_amsgrad_heterogeneous_py(
    at::Tensor  params,
    at::Tensor  exp_avg,
    at::Tensor  exp_avg_sq,
    at::Tensor  exp_avg_sq_max,
    at::Tensor  grads,
    float       lr_base,
    float       lr_scale,
    float       beta1,
    float       beta2,
    float       bc1,
    float       bc2,
    float       eps,
    float       weight_decay,
    float       clip_scale,
    float       fp8_grad_scale,
    int         grad_dtype,
    int         sm_version,
    at::Tensor  master_params_opt)
{
    check_bf16(params,  "params");
    check_fp32(exp_avg,       "exp_avg");
    check_fp32(exp_avg_sq,    "exp_avg_sq");
    check_fp32(exp_avg_sq_max, "exp_avg_sq_max");
    TORCH_CHECK(grads.is_cuda() && grads.is_contiguous(), "grads must be contiguous CUDA tensor");

    const size_t n_elems = (size_t)params.numel();
    TORCH_CHECK((size_t)exp_avg.numel()       == n_elems, "exp_avg numel mismatch");
    TORCH_CHECK((size_t)exp_avg_sq.numel()    == n_elems, "exp_avg_sq numel mismatch");
    TORCH_CHECK((size_t)exp_avg_sq_max.numel() == n_elems, "exp_avg_sq_max numel mismatch");
    TORCH_CHECK(lr_base > 0.f, "lr_base must be positive");
    TORCH_CHECK(lr_scale > 0.f, "lr_scale must be positive");
    TORCH_CHECK(bc1 > 0.f && bc2 > 0.f, "bias corrections must be positive");
    TORCH_CHECK(eps > 0.f, "eps must be positive");
    TORCH_CHECK(clip_scale > 0.f, "clip_scale must be positive");

    float* master_ptr = nullptr;
    if (master_params_opt.defined() && master_params_opt.numel() > 0) {
        check_fp32(master_params_opt, "master_params");
        TORCH_CHECK(master_params_opt.is_contiguous());
        TORCH_CHECK((size_t)master_params_opt.numel() == n_elems, "master_params numel mismatch");
        master_ptr = master_params_opt.data_ptr<float>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_adamw_amsgrad_heterogeneous(
        reinterpret_cast<__nv_bfloat16*>(params.data_ptr<at::BFloat16>()),
        master_ptr,
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        exp_avg_sq_max.data_ptr<float>(),
        grads.data_ptr(),
        n_elems,
        lr_base, lr_scale,
        beta1, beta2, bc1, bc2, eps, weight_decay,
        clip_scale, fp8_grad_scale, grad_dtype,
        sm_version, stream);
}

// ---------------------------------------------------------------------------
// pcie_tree_reduce_step binding  (#146)
// ---------------------------------------------------------------------------

void pcie_tree_reduce_step_py(at::Tensor accum_buf,
                                at::Tensor recv_buf,
                                int sm_version)
{
    check_bf16(accum_buf, "accum_buf");
    check_bf16(recv_buf,  "recv_buf");
    TORCH_CHECK(accum_buf.numel() == recv_buf.numel(),
                "accum_buf/recv_buf numel mismatch");
    TORCH_CHECK(accum_buf.numel() % 8 == 0,
                "numel must be divisible by 8");
    TORCH_CHECK(accum_buf.numel() > 0, "accum_buf must be non-empty");
    TORCH_CHECK(accum_buf.device() == recv_buf.device(),
                "accum_buf and recv_buf must be on the same CUDA device");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_pcie_tree_reduce_step(
        reinterpret_cast<__nv_bfloat16*>(accum_buf.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(recv_buf.data_ptr<at::BFloat16>()),
        (size_t)accum_buf.numel(), sm_version, stream);
}

// PYBIND11_MODULE
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "DeepSpeed hetero_reduce: fused BF16 reduce-scatter + SwiGLU-LN + "
              "RoPE + PCIe allreduce + tier activation offload kernels "
              "for heterogeneous GPU clusters (SM 8.6 / 9.0 / 12.0).";

    m.def("fused_bf16_reduce",
          &fused_bf16_reduce_py,
          "Fused BF16→FP32 reduce + FP32→BF16 writeback across multiple tensors.\n"
          "Args:\n"
          "  output     (Tensor BF16): in-place reduction destination\n"
          "  inputs     (List[Tensor BF16]): tensors to reduce\n"
          "  sm_version (int): SM version of active device (86, 90, 120, …)",
          py::arg("output"),
          py::arg("inputs"),
          py::arg("sm_version") = 86);

    m.def("hetero_reduce_scatter",
          &hetero_reduce_scatter_py,
          "Heterogeneous reduce-scatter: reduces all inputs but writes only the\n"
          "local shard [shard_offset, shard_offset + shard_count) to output.\n"
          "Args:\n"
          "  output       (Tensor BF16): shard output buffer [shard_count]\n"
          "  inputs       (List[Tensor BF16]): full-length input gradient tensors\n"
          "  shard_offset (int): starting element index in full tensor\n"
          "  shard_count  (int): number of elements to reduce and write\n"
          "  sm_version   (int): SM version of active device",
          py::arg("output"),
          py::arg("inputs"),
          py::arg("shard_offset"),
          py::arg("shard_count"),
          py::arg("sm_version") = 86);

    m.def("compute_shard_ranges",
          &compute_shard_ranges_py,
          "Compute non-uniform shard ranges for heterogeneous GPU tiers.\n"
          "Returns List[Tuple[offset, count]] with one entry per tier.\n"
          "Weight: SM12.0=4, SM9.0=3, SM8.6=1.\n"
          "Args:\n"
          "  sm_versions (List[int]): per-tier SM versions\n"
          "  total_elems (int): total BF16 elements in gradient tensor",
          py::arg("sm_versions"),
          py::arg("total_elems"));

    m.def("fused_swiglu_ln",
          &fused_swiglu_ln_py,
          "Fused SwiGLU activation + RMS LayerNorm.\n"
          "Args:\n"
          "  output     (Tensor BF16  [B, H]): output buffer\n"
          "  gate_proj  (Tensor BF16  [B, H]): gate projection\n"
          "  up_proj    (Tensor BF16  [B, H]): up   projection\n"
          "  ln_weight  (Tensor FP32  [H])   : RMSNorm scale\n"
          "  eps        (float)               : RMSNorm epsilon\n"
          "  sm_version (int)                 : 86, 90, or 120",
          py::arg("output"),
          py::arg("gate_proj"),
          py::arg("up_proj"),
          py::arg("ln_weight"),
          py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // fused_rope_hetero
    // -----------------------------------------------------------------------
    m.def("rope_cache",
          &rope_cache_py,
          "Precompute RoPE cos/sin cache on device.\n"
          "Args:\n"
          "  cos_cache  (Tensor FP32 [S, D/2]): output cosine table\n"
          "  sin_cache  (Tensor FP32 [S, D/2]): output sine table\n"
          "  seq_len    (int): sequence length\n"
          "  head_dim   (int): full head dimension\n"
          "  base       (float): RoPE base, default 10000.0\n"
          "  pos_offset (int): global position offset for packed seqs",
          py::arg("cos_cache"),
          py::arg("sin_cache"),
          py::arg("seq_len"),
          py::arg("head_dim"),
          py::arg("base") = 10000.f,
          py::arg("pos_offset") = 0);

    m.def("fused_rope_hetero",
          &fused_rope_hetero_py,
          "Fused RoPE for heterogeneous head counts.\n"
          "Args:\n"
          "  output     (Tensor BF16 [B, S, H, D]): output (may alias input)\n"
          "  input      (Tensor BF16 [B, S, H, D]): query or key tensor\n"
          "  cos_cache  (Tensor FP32 [S, D/2])    : precomputed cosines\n"
          "  sin_cache  (Tensor FP32 [S, D/2])    : precomputed sines\n"
          "  neox_style (bool): True=Llama/NeoX, False=GPT-J interleaved\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("output"),
          py::arg("input"),
          py::arg("cos_cache"),
          py::arg("sin_cache"),
          py::arg("neox_style") = true,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // pcie_adaptive_allreduce
    // -----------------------------------------------------------------------
    m.def("pcie_gradient_pack",
          &pcie_gradient_pack_py,
          "Gather non-contiguous gradient shards into a flat BF16 bucket.\n"
          "Args:\n"
          "  bucket     (Tensor BF16 [bucket_elems]): flat output bucket\n"
          "  chunks     (List[Tuple[Tensor, int, int]]): (tensor, offset, length) per shard\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("bucket"),
          py::arg("chunks"),
          py::arg("sm_version") = 86);

    m.def("pcie_ring_reduce",
          &pcie_ring_reduce_py,
          "PCIe ring-allreduce reduce phase: dst += src (BF16, in-place).\n"
          "Args:\n"
          "  dst        (Tensor BF16): local accumulator (modified in-place)\n"
          "  src        (Tensor BF16): incoming peer gradient bucket\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("dst"),
          py::arg("src"),
          py::arg("sm_version") = 86);

    m.def("pcie_allreduce_finalise",
          &pcie_allreduce_finalise_py,
          "Divide allreduce sum by world_size and write BF16 output.\n"
          "Args:\n"
          "  out        (Tensor BF16): output buffer\n"
          "  src        (Tensor BF16): sum buffer\n"
          "  world_size (int): number of participating GPUs\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("out"),
          py::arg("src"),
          py::arg("world_size"),
          py::arg("sm_version") = 86);

    m.def("pcie_bucket_size",
          &pcie_bucket_size_py,
          "Compute recommended PCIe gradient bucket size in bytes.\n"
          "Args:\n"
          "  pcie_bw_gbps (float): measured or estimated PCIe bandwidth in GB/s\n"
          "Returns: int (bucket size in bytes)",
          py::arg("pcie_bw_gbps") = 32.f);

    // -----------------------------------------------------------------------
    // tier_activation_offload
    // -----------------------------------------------------------------------
    m.def("activation_pack",
          &activation_pack_py,
          "Pack activation tensors into a flat BF16 offload buffer.\n"
          "Args:\n"
          "  output     (Tensor BF16 [N * tensor_elems]): flat output buffer\n"
          "  inputs     (List[Tensor BF16]): activation tensors to pack\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("output"),
          py::arg("inputs"),
          py::arg("sm_version") = 86);

    m.def("activation_unpack",
          &activation_unpack_py,
          "Unpack a flat BF16 buffer back to individual activation tensors.\n"
          "Args:\n"
          "  outputs    (List[Tensor BF16]): destination activation tensors\n"
          "  flat       (Tensor BF16 [N * tensor_elems]): flat source buffer\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("outputs"),
          py::arg("flat"),
          py::arg("sm_version") = 86);

    m.def("quantise_bf16_to_int8",
          &quantise_bf16_to_int8_py,
          "Block-wise INT8 quantisation of BF16 activation buffer.\n"
          "Tile size = 128 elements, scale = absmax / 127 per tile.\n"
          "Args:\n"
          "  output (Tensor Int8  [N]): quantised output\n"
          "  scales (Tensor FP32  [ceil(N/128)]): per-tile scales\n"
          "  input  (Tensor BF16  [N]): input activations",
          py::arg("output"),
          py::arg("scales"),
          py::arg("input"));

    m.def("dequantise_int8_to_bf16",
          &dequantise_int8_to_bf16_py,
          "Block-wise INT8 dequantisation to BF16.\n"
          "Args:\n"
          "  output (Tensor BF16  [N]): dequantised output\n"
          "  input  (Tensor Int8  [N]): quantised input\n"
          "  scales (Tensor FP32  [ceil(N/128)]): per-tile scales",
          py::arg("output"),
          py::arg("input"),
          py::arg("scales"));

    m.def("compute_offload_budget",
          &compute_offload_budget_py,
          "Compute activation offload budget for a GPU tier.\n"
          "Args:\n"
          "  total_act_bytes  (int): total activation bytes required\n"
          "  vram_free_bytes  (int): current free VRAM on this device\n"
          "  headroom_frac    (float): fraction of free VRAM to keep unused\n"
          "Returns: int (bytes to offload, 0 if activations fit in VRAM)",
          py::arg("total_act_bytes"),
          py::arg("vram_free_bytes"),
          py::arg("headroom_frac") = 0.1f);
    // -----------------------------------------------------------------------
    // fused_layernorm_residual  (#110)
    // -----------------------------------------------------------------------
    m.def("fused_layernorm_residual",
          &fused_layernorm_residual_py,
          "Fused residual add + RMS LayerNorm (pre-LN Llama/Mistral style).\n"
          "residual_i += input_i;  output_i = rmsnorm(residual_i) * weight.\n"
          "Args:\n"
          "  output     (Tensor BF16  [B, H]): LN output\n"
          "  residual   (Tensor BF16  [B, H]): residual stream, updated in-place\n"
          "  input      (Tensor BF16  [B, H]): new sub-layer contribution\n"
          "  ln_weight  (Tensor FP32  [H])   : RMSNorm gamma scale\n"
          "  eps        (float)               : RMSNorm epsilon (default 1e-6)\n"
          "  sm_version (int)                 : 86, 90, or 120",
          py::arg("output"),
          py::arg("residual"),
          py::arg("input"),
          py::arg("ln_weight"),
          py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // cross_entropy_tp  (#110)
    // -----------------------------------------------------------------------
    m.def("cross_entropy_tp_forward",
          &cross_entropy_tp_forward_py,
          "Phase-1 TP cross-entropy: local (max, sum_exp, label_logit).\n"
          "Returns Tuple[Tensor FP32 [B], Tensor FP32 [B], Tensor FP32 [B]].",
          py::arg("logits"),
          py::arg("labels"),
          py::arg("shard_offset") = 0,
          py::arg("sm_version") = 86);

    m.def("cross_entropy_tp_loss",
          &cross_entropy_tp_loss_py,
          "Phase-2 TP cross-entropy: per-sample CE loss from reduced scalars.\n"
          "Returns: loss (Tensor FP32 [B])",
          py::arg("global_max"),
          py::arg("global_sum_exp"),
          py::arg("global_logit"));

    m.def("cross_entropy_tp_backward",
          &cross_entropy_tp_backward_py,
          "TP cross-entropy backward: softmax gradient w.r.t. local logit shard.",
          py::arg("d_logits"),
          py::arg("logits"),
          py::arg("labels"),
          py::arg("global_max"),
          py::arg("log_sum_exp"),
          py::arg("shard_offset") = 0,
          py::arg("inv_batch") = 1.f,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // cross_entropy_tp heterogeneous vocab partition  (#141)
    // -----------------------------------------------------------------------
    m.def("compute_hetero_vocab_partition",
          &compute_hetero_vocab_partition_py,
          "Compute per-rank VocabPartition for a non-uniform TP vocab split.\n"
          "Returns a list of dicts {v_local, shard_offset, tp_size, rank}.\n"
          "\n"
          "Token counts are proportional to SM weights (SM12.0→4, SM9.0→3,\n"
          "SM8.6→1) and aligned to 8 elements (kVecBF16).  The last rank\n"
          "absorbs any residual.\n"
          "\n"
          "Args:\n"
          "  sm_versions (List[int]): SM version per rank, e.g. [86, 90, 120]\n"
          "  vocab_size  (int):       full vocabulary size V\n",
          py::arg("sm_versions"),
          py::arg("vocab_size"));

    m.def("cross_entropy_tp_forward_hetero",
          &cross_entropy_tp_forward_hetero_py,
          "Phase-1 heterogeneous TP cross-entropy forward (issue #141).\n"
          "Pass A: compute local (max, sum_exp, label_logit) for a non-uniform\n"
          "vocab shard of width v_local starting at shard_offset.\n"
          "\n"
          "Pass B (caller): AllReduce across TP ranks, then call\n"
          "cross_entropy_tp_loss as usual.\n"
          "\n"
          "Returns Tuple[Tensor FP32 [B], Tensor FP32 [B], Tensor FP32 [B]]\n"
          "        (local_max, local_sum_exp, local_logit).",
          py::arg("logits"),
          py::arg("labels"),
          py::arg("v_local"),
          py::arg("shard_offset") = 0,
          py::arg("sm_version") = 86);

    m.def("cross_entropy_tp_backward_hetero",
          &cross_entropy_tp_backward_hetero_py,
          "Backward pass for heterogeneous TP cross-entropy (issue #141).\n"
          "Computes softmax gradient w.r.t. a non-uniform vocab shard.",
          py::arg("d_logits"),
          py::arg("logits"),
          py::arg("labels"),
          py::arg("global_max"),
          py::arg("log_sum_exp"),
          py::arg("v_local"),
          py::arg("shard_offset") = 0,
          py::arg("inv_batch") = 1.f,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // fused_adam_heterogeneous — per-tier LR-scaled Adam for A6000/H100/Blackwell
    // -----------------------------------------------------------------------
    m.def("fused_adam_heterogeneous",
          &fused_adam_heterogeneous_py,
          "Fused Adam optimizer with per-tier learning-rate scaling.\n"
          "Applies the AdamW update (decoupled weight decay) with an effective\n"
          "LR of lr_base × lr_scale.  BF16 params/grads, FP32 moments.\n"
          "Optional FP32 master-weight copy kept when master_params is non-empty.\n"
          "\n"
          "Args:\n"
          "  params        (Tensor BF16  [N]): working parameters (updated in-place)\n"
          "  exp_avg       (Tensor FP32  [N]): first-moment  buffer (updated in-place)\n"
          "  exp_avg_sq    (Tensor FP32  [N]): second-moment buffer (updated in-place)\n"
          "  grads         (Tensor BF16  [N]): gradient tensor for this step\n"
          "  lr_base       (float): base learning rate\n"
          "  lr_scale      (float): per-tier LR multiplier (use hetero_adam_lr_scale)\n"
          "  beta1         (float): Adam β₁, default 0.9\n"
          "  beta2         (float): Adam β₂, default 0.999\n"
          "  bc1           (float): bias-correction-1 = 1/(1−β₁^step)\n"
          "  bc2           (float): bias-correction-2 = 1/(1−β₂^step)\n"
          "  eps           (float): Adam ε, default 1e-8\n"
          "  weight_decay  (float): decoupled weight-decay coefficient, default 0.0\n"
          "  sm_version    (int)  : 86, 90, or 120\n"
          "  master_params (Tensor FP32 [N]): optional FP32 master copy; pass an\n"
          "                 empty tensor (torch.Tensor()) to disable",
          py::arg("params"),
          py::arg("exp_avg"),
          py::arg("exp_avg_sq"),
          py::arg("grads"),
          py::arg("lr_base"),
          py::arg("lr_scale"),
          py::arg("beta1")        = 0.9f,
          py::arg("beta2")        = 0.999f,
          py::arg("bc1")          = 1.f,
          py::arg("bc2")          = 1.f,
          py::arg("eps")          = 1e-8f,
          py::arg("weight_decay") = 0.f,
          py::arg("sm_version")   = 86,
          py::arg("master_params") = at::Tensor());

    m.def("hetero_adam_lr_scale",
          &hetero_adam_lr_scale_py,
          "Return the default per-tier LR scale for a given SM version.\n"
          "SM12.0 (Blackwell) → 4.0, SM9.0 (H100) → 3.0, SM8.6 (A6000) → 1.0.\n"
          "Args:\n"
          "  sm_version (int): 86, 90, or 120\n"
          "Returns: float",
          py::arg("sm_version"));

    // -----------------------------------------------------------------------
    // Additional utility APIs
    // -----------------------------------------------------------------------
    m.def("hetero_bucket_size_elems",
          &hetero_bucket_size_elems_py,
          "Return policy-recommended gradient bucket size (BF16 elements) for SM.\n"
          "SM12.0: 2M, SM9.0: 4M, SM8.6: 512K.\n"
          "Args:\n"
          "  sm_version (int): 86, 90, or 120\n"
          "Returns: int (elements)",
          py::arg("sm_version") = 86);

    m.def("compute_adaptive_chunk_size",
          &compute_adaptive_chunk_size_py,
          "Compute adaptive ring-allreduce chunk size in bytes.\n"
          "Targets kTargetOverlapMs (5ms) of PCIe transfer per ring step.\n"
          "Args:\n"
          "  pcie_bw_gbps (float): measured PCIe bandwidth in GB/s\n"
          "Returns: int (chunk size in bytes, 16-byte aligned)",
          py::arg("pcie_bw_gbps") = 32.f);

    m.def("probe_pcie_bandwidth",
          &probe_pcie_bandwidth_py,
          "Probe PCIe bandwidth between two CUDA devices.\n"
          "Sends 4 MB test transfer, times it with CUDA events.\n"
          "Args:\n"
          "  src_device (int): source CUDA device ordinal\n"
          "  dst_device (int): destination CUDA device ordinal\n"
          "Returns: float (measured bandwidth in GB/s)",
          py::arg("src_device"),
          py::arg("dst_device"));

    m.def("pcie_ring_reduce_step",
          &pcie_ring_reduce_step_py,
          "Single double-buffered ring-allreduce reduce step.\n"
          "accum_buf += recv_buf (BF16 → FP32 accumulation → BF16).\n"
          "Args:\n"
          "  accum_buf  (Tensor BF16): local accumulator (modified in-place)\n"
          "  recv_buf   (Tensor BF16): received chunk from ring peer\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("accum_buf"),
          py::arg("recv_buf"),
          py::arg("sm_version") = 86);


    // -----------------------------------------------------------------------
    // Extended LayerNorm + RoPE cacheless + FP8 grad norm
    // -----------------------------------------------------------------------
    m.def("fused_layernorm_residual_ex",
          &fused_layernorm_residual_ex_py,
          "Extended fused residual + LN: full LN or RMSNorm, optional bias, optional FP32 out.\n"
          "Args:\n"
          "  output       (Tensor BF16 [B,H]): LN output\n"
          "  residual     (Tensor BF16 [B,H]): residual stream (updated in-place)\n"
          "  input        (Tensor BF16 [B,H]): new contribution\n"
          "  ln_weight    (Tensor FP32 [H]):   scale\n"
          "  eps          (float): epsilon\n"
          "  full_ln      (bool): True=full LayerNorm (Welford), False=RMSNorm\n"
          "  sm_version   (int): 86, 90, 120\n"
          "  bias         (Tensor BF16 [H] or None): optional bias\n"
          "  output_fp32  (Tensor FP32 [B,H] or None): optional FP32 output",
          py::arg("output"),
          py::arg("residual"),
          py::arg("input"),
          py::arg("ln_weight"),
          py::arg("eps") = 1e-6f,
          py::arg("full_ln") = false,
          py::arg("sm_version") = 86,
          py::arg("bias") = py::none(),
          py::arg("output_fp32") = py::none());

    m.def("fused_rope_cacheless",
          &fused_rope_cacheless_py,
          "RoPE with on-the-fly sin/cos computation (no precomputed cache).\n"
          "Use for very long sequences where cache exceeds L2.\n"
          "Args:\n"
          "  output     (Tensor BF16 [B,S,H,D])\n"
          "  input      (Tensor BF16 [B,S,H,D])\n"
          "  base       (float): RoPE base, default 10000.0\n"
          "  pos_offset (int): global position offset for packed sequences\n"
          "  neox_style (bool): True=Llama/NeoX, False=GPT-J interleaved\n"
          "  sm_version (int): 86, 90, 120",
          py::arg("output"),
          py::arg("input"),
          py::arg("base") = 10000.f,
          py::arg("pos_offset") = 0,
          py::arg("neox_style") = true,
          py::arg("sm_version") = 86);

    m.def("grad_norm_sq_fp8",
          &grad_norm_sq_fp8_py,
          "Accumulate gradient L2 norm squared for FP8-E4M3 gradients.\n"
          "Uses Kahan compensated summation for numerical accuracy.\n"
          "Args:\n"
          "  grads          (Tensor Int8/Byte [N]): FP8-E4M3 gradient buffer\n"
          "  norm_sq_accum  (Tensor FP32 [1]): accumulator (add to, not reset)\n"
          "  fp8_scale      (float): per-tensor FP8 scale factor\n"
          "  sm_version     (int): 86, 90, 120",
          py::arg("grads"),
          py::arg("norm_sq_accum"),
          py::arg("fp8_scale") = 1.f,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // hetero_ring_allreduce — 5-GPU 2-NUMA PCIe ring allreduce
    //
    // Exposed API:
    //   hetero_ring_reduce_step(accum_buf, recv_buf, sm_version) -> None
    //   hetero_ring_gather_step(output, recv_buf, sm_version)    -> None
    //   hetero_ring_intra_numa_chunk_bytes()  -> int
    //   hetero_ring_cross_numa_chunk_bytes()  -> int
    //   hetero_ring_max_chunk_bytes()         -> int
    //   hetero_ring_sm_block_size(sm_version) -> int
    // -----------------------------------------------------------------------

    m.def("hetero_ring_reduce_step",
          [](at::Tensor accum_buf, at::Tensor recv_buf, int sm_version) {
              check_bf16(accum_buf, "accum_buf");
              check_bf16(recv_buf,  "recv_buf");
              TORCH_CHECK(accum_buf.numel() == recv_buf.numel(),
                          "accum_buf and recv_buf must have the same number of elements");
              TORCH_CHECK(accum_buf.numel() > 0, "accum_buf must be non-empty");
              TORCH_CHECK(accum_buf.device() == recv_buf.device(),
                          "accum_buf and recv_buf must be on the same CUDA device");
              const size_t n = (size_t)accum_buf.numel();
              cudaStream_t stream = at::cuda::getCurrentCUDAStream();
              launch_hetero_ring_reduce_step(
                  reinterpret_cast<__nv_bfloat16*>(accum_buf.data_ptr()),
                  reinterpret_cast<const __nv_bfloat16*>(recv_buf.data_ptr()),
                  n, sm_version, stream);
          },
          "Single heterogeneous ring reduce-scatter step.\n"
          "accum_buf[i] += recv_buf[i]  (BF16 -> FP32 accumulation -> BF16).\n"
          "Dispatches SM-specialised kernel: SM8.6 (A6000), SM9.0 (H100),\n"
          "or SM12.0 (Blackwell, cp.async double-buffered).\n"
          "\n"
          "Args:\n"
          "  accum_buf  (Tensor BF16 [N]): local accumulator (modified in-place)\n"
          "  recv_buf   (Tensor BF16 [N]): received chunk from ring predecessor\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("accum_buf"),
          py::arg("recv_buf"),
          py::arg("sm_version") = 86);

    m.def("hetero_ring_gather_step",
          [](at::Tensor output, at::Tensor recv_buf, int sm_version) {
              check_bf16(output,   "output");
              check_bf16(recv_buf, "recv_buf");
              TORCH_CHECK(output.numel() == recv_buf.numel(),
                          "output and recv_buf must have the same number of elements");
              TORCH_CHECK(output.numel() > 0, "output must be non-empty");
              TORCH_CHECK(output.device() == recv_buf.device(),
                          "output and recv_buf must be on the same CUDA device");
              const size_t n = (size_t)output.numel();
              cudaStream_t stream = at::cuda::getCurrentCUDAStream();
              launch_hetero_ring_gather_step(
                  reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                  reinterpret_cast<const __nv_bfloat16*>(recv_buf.data_ptr()),
                  n, sm_version, stream);
          },
          "Single heterogeneous ring all-gather step.\n"
          "output[i] = recv_buf[i]  (128-bit vectorised copy, no accumulation).\n"
          "Used in the all-gather phase after reduce-scatter completes.\n"
          "\n"
          "Args:\n"
          "  output     (Tensor BF16 [N]): destination buffer\n"
          "  recv_buf   (Tensor BF16 [N]): fully-reduced chunk from ring predecessor\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("output"),
          py::arg("recv_buf"),
          py::arg("sm_version") = 86);

    m.def("hetero_ring_intra_numa_chunk_bytes",
          []() -> int64_t { return (int64_t)hetero_ring_intra_numa_chunk_bytes(); },
          "Intra-NUMA chunk size in bytes (4 MB).\n"
          "Used for ring steps within the same NUMA node (~32 GB/s PCIe 4.0 x16).\n"
          "Returns: int");

    m.def("hetero_ring_cross_numa_chunk_bytes",
          []() -> int64_t { return (int64_t)hetero_ring_cross_numa_chunk_bytes(); },
          "Cross-NUMA chunk size in bytes (2 MB).\n"
          "Used for ring steps crossing NUMA domains (~16 GB/s cross-switch PCIe).\n"
          "Returns: int");

    m.def("hetero_ring_max_chunk_bytes",
          []() -> int64_t { return (int64_t)hetero_ring_max_chunk_bytes(); },
          "Maximum chunk size in bytes (4 MB = intra-NUMA chunk).\n"
          "Allocate ping/pong receive buffers with at least this many bytes.\n"
          "Returns: int");

    m.def("hetero_ring_sm_block_size",
          &hetero_ring_sm_block_size,
          "Thread-block size for hetero ring reduce/gather kernels.\n"
          "SM8.6 (A6000): 128 threads (fewer SMs -> smaller blocks)\n"
          "SM9.0  (H100): 256 threads\n"
          "SM12.0 (Blackwell): 512 threads (widest warp scheduler)\n"
          "\n"
          "Args:\n"
          "  sm_version (int): 86, 90, or 120\n"
          "Returns: int (thread-block size)",
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // fused_gradient_allreduce — INT8 compressed ring allreduce  (#146)
    // -----------------------------------------------------------------------
    m.def("gradient_compress",
          &gradient_compress_py,
          "Compress BF16 gradient to INT8 with per-block FP32 scales.\n"
          "Block size = 256 elements.\n"
          "Args:\n"
          "  output_int8  (Tensor Int8  [N]): compressed output\n"
          "  output_scale (Tensor FP32  [ceil(N/256)]): per-block scales\n"
          "  input        (Tensor BF16  [N]): input gradient\n"
          "  sm_version   (int): 86, 90, or 120",
          py::arg("output_int8"),
          py::arg("output_scale"),
          py::arg("input"),
          py::arg("sm_version") = 86);

    m.def("int8_ring_reduce_step",
          &int8_ring_reduce_step_py,
          "Fused INT8 ring-allreduce accumulation step.\n"
          "Dequantises dst and src, sums, re-quantises in-place.\n"
          "Args:\n"
          "  dst_int8  (Tensor Int8  [N]): accumulator (modified in-place)\n"
          "  dst_scale (Tensor FP32  [ceil(N/256)]): dst per-block scales\n"
          "  src_int8  (Tensor Int8  [N]): received chunk from ring peer\n"
          "  src_scale (Tensor FP32  [ceil(N/256)]): src per-block scales\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("dst_int8"),
          py::arg("dst_scale"),
          py::arg("src_int8"),
          py::arg("src_scale"),
          py::arg("sm_version") = 86);

    m.def("gradient_decompress",
          &gradient_decompress_py,
          "Decompress INT8 + per-block scales back to BF16.\n"
          "Args:\n"
          "  output    (Tensor BF16  [N]): decompressed output\n"
          "  int8_data (Tensor Int8  [N]): compressed data\n"
          "  scale_buf (Tensor FP32  [ceil(N/256)]): per-block scales\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("output"),
          py::arg("int8_data"),
          py::arg("scale_buf"),
          py::arg("sm_version") = 86);

    m.def("gradient_allreduce_finalise",
          &gradient_allreduce_finalise_py,
          "Apply averaging (divide scales by world_size) after ring reduce.\n"
          "Args:\n"
          "  scale_buf   (Tensor FP32): per-block scales (modified in-place)\n"
          "  n_elems     (int): total gradient elements\n"
          "  world_size  (int): number of participating ranks",
          py::arg("scale_buf"),
          py::arg("n_elems"),
          py::arg("world_size"));

    m.def("gradient_compress_bytes",
          &gradient_compress_bytes_py,
          "Return INT8 staging buffer size (bytes) for n_elems gradient elements.\n"
          "Args:\n"
          "  n_elems (int): number of BF16 gradient elements\n"
          "Returns: int (bytes)",
          py::arg("n_elems"));

    m.def("gradient_scale_bytes",
          &gradient_scale_bytes_py,
          "Return per-block scale buffer size (bytes) for n_elems gradient elements.\n"
          "Args:\n"
          "  n_elems (int): number of BF16 gradient elements\n"
          "Returns: int (bytes)",
          py::arg("n_elems"));

    // -----------------------------------------------------------------------
    // fused_swiglu_ln forward-save & backward  (#146)
    // -----------------------------------------------------------------------
    m.def("fused_swiglu_ln_fwd_save",
          &fused_swiglu_ln_fwd_save_py,
          "Forward SwiGLU + RMSNorm, saving rms_inv for backward.\n"
          "Args:\n"
          "  output      (Tensor BF16  [B, H]): fused output\n"
          "  rms_inv_out (Tensor FP32  [B]):    saved 1/rms per row\n"
          "  gate_proj   (Tensor BF16  [B, H]): gate projection\n"
          "  up_proj     (Tensor BF16  [B, H]): up projection\n"
          "  ln_weight   (Tensor FP32  [H]):    RMSNorm scale\n"
          "  eps         (float): RMSNorm epsilon\n"
          "  sm_version  (int): 86, 90, or 120",
          py::arg("output"),
          py::arg("rms_inv_out"),
          py::arg("gate_proj"),
          py::arg("up_proj"),
          py::arg("ln_weight"),
          py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);

    m.def("fused_swiglu_ln_backward",
          &fused_swiglu_ln_backward_py,
          "Backward pass for fused SwiGLU + RMSNorm.\n"
          "d_ln_weight must be zeroed before the first call; gradients accumulate\n"
          "via atomicAdd across the batch.\n"
          "Args:\n"
          "  d_gate      (Tensor BF16  [B, H]): gradient w.r.t. gate_proj\n"
          "  d_up        (Tensor BF16  [B, H]): gradient w.r.t. up_proj\n"
          "  d_ln_weight (Tensor FP32  [H]):    gradient w.r.t. ln_weight (accumulated)\n"
          "  d_output    (Tensor BF16  [B, H]): upstream gradient\n"
          "  gate_proj   (Tensor BF16  [B, H]): saved gate projection from forward\n"
          "  up_proj     (Tensor BF16  [B, H]): saved up projection from forward\n"
          "  ln_weight   (Tensor FP32  [H]):    RMSNorm scale\n"
          "  rms_inv_buf (Tensor FP32  [B]):    saved 1/rms from fwd_save\n"
          "  eps         (float): RMSNorm epsilon\n"
          "  sm_version  (int): 86, 90, or 120",
          py::arg("d_gate"),
          py::arg("d_up"),
          py::arg("d_ln_weight"),
          py::arg("d_output"),
          py::arg("gate_proj"),
          py::arg("up_proj"),
          py::arg("ln_weight"),
          py::arg("rms_inv_buf"),
          py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // grad_norm_sq — BF16 gradient L2 norm squared  (#146)
    // -----------------------------------------------------------------------
    m.def("grad_norm_sq",
          &grad_norm_sq_py,
          "Accumulate gradient L2 norm squared for BF16 gradients.\n"
          "Caller must zero norm_sq_accum before the first call per step.\n"
          "Args:\n"
          "  grads         (Tensor BF16 [N]): gradient buffer\n"
          "  norm_sq_accum (Tensor FP32 [1]): accumulator (add to, not reset)\n"
          "  sm_version    (int): 86, 90, or 120",
          py::arg("grads"),
          py::arg("norm_sq_accum"),
          py::arg("sm_version") = 86);

    // -----------------------------------------------------------------------
    // fused_adamw_amsgrad_heterogeneous — AMSGrad + gradient clipping  (#146)
    // -----------------------------------------------------------------------
    m.def("fused_adamw_amsgrad_heterogeneous",
          &fused_adamw_amsgrad_heterogeneous_py,
          "Fused AdamW with AMSGrad, per-tier LR scaling, gradient clipping,\n"
          "and optional FP8-E4M3 gradient input.\n"
          "Args:\n"
          "  params          (Tensor BF16 [N]): working parameters\n"
          "  exp_avg         (Tensor FP32 [N]): first moment\n"
          "  exp_avg_sq      (Tensor FP32 [N]): second moment\n"
          "  exp_avg_sq_max  (Tensor FP32 [N]): AMSGrad running max of v_hat\n"
          "  grads           (Tensor [N]): gradients (BF16 or FP8 per grad_dtype)\n"
          "  lr_base         (float): base learning rate\n"
          "  lr_scale        (float): per-tier LR multiplier\n"
          "  beta1           (float): Adam beta1\n"
          "  beta2           (float): Adam beta2\n"
          "  bc1             (float): bias-correction-1 = 1/(1-beta1^step)\n"
          "  bc2             (float): bias-correction-2 = 1/(1-beta2^step)\n"
          "  eps             (float): Adam epsilon\n"
          "  weight_decay    (float): decoupled weight-decay\n"
          "  clip_scale      (float): gradient clip scale, min(1, clip_norm/grad_norm)\n"
          "  fp8_grad_scale  (float): FP8 per-tensor scale (1.0 for BF16)\n"
          "  grad_dtype      (int): 0=BF16, 1=FP8-E4M3\n"
          "  sm_version      (int): 86, 90, or 120\n"
          "  master_params   (Tensor FP32 [N]): optional FP32 master copy",
          py::arg("params"),
          py::arg("exp_avg"),
          py::arg("exp_avg_sq"),
          py::arg("exp_avg_sq_max"),
          py::arg("grads"),
          py::arg("lr_base"),
          py::arg("lr_scale"),
          py::arg("beta1")          = 0.9f,
          py::arg("beta2")          = 0.999f,
          py::arg("bc1")            = 1.f,
          py::arg("bc2")            = 1.f,
          py::arg("eps")            = 1e-8f,
          py::arg("weight_decay")   = 0.f,
          py::arg("clip_scale")     = 1.f,
          py::arg("fp8_grad_scale") = 1.f,
          py::arg("grad_dtype")     = 0,
          py::arg("sm_version")     = 86,
          py::arg("master_params")  = at::Tensor());

    // -----------------------------------------------------------------------
    // pcie_tree_reduce_step — tree-topology reduce  (#146)
    // -----------------------------------------------------------------------
    m.def("pcie_tree_reduce_step",
          &pcie_tree_reduce_step_py,
          "Single tree-topology reduce step: accum_buf += recv_buf (BF16).\n"
          "Args:\n"
          "  accum_buf  (Tensor BF16 [N]): local accumulator (modified in-place)\n"
          "  recv_buf   (Tensor BF16 [N]): received chunk from child node\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("accum_buf"),
          py::arg("recv_buf"),
          py::arg("sm_version") = 86);


}