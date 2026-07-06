// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * binding.cpp
 *
 * PyTorch / pybind11 bindings for the hetero_reduce CUDA kernels.
 *
 * Expanded in this revision:
 *   - probe_pcie_bandwidth           (bandwidth measurement between devices)
 *   - compute_adaptive_chunk_size    (adaptive ring-allreduce chunk sizing)
 *   - hetero_bucket_size_elems       (per-SM bucket size query)
 *   - gradient_compress              (BF16 → INT8 compressed gradient)
 *   - int8_ring_reduce_step          (INT8 ring-reduce accumulation)
 *   - gradient_decompress            (INT8 → BF16 gradient)
 *   - gradient_allreduce_finalise    (scale divide by world_size)
 *   - fused_gradient_allreduce       (high-level 3-phase INT8 all-reduce)
 *   - gradient_compress_bytes/scale_bytes (buffer sizing utilities)
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

static void check_int8(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Char,
                name, " must be Int8, got ", t.scalar_type());
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

    TORCH_CHECK(hidden % 8 == 0, "hidden must be divisible by 8, got ", hidden);
    TORCH_CHECK(gate_proj.size(0) == batch && gate_proj.size(1) == hidden, "gate_proj shape mismatch");
    TORCH_CHECK(up_proj.size(0)   == batch && up_proj.size(1)   == hidden, "up_proj shape mismatch");
    TORCH_CHECK(ln_weight.numel() == hidden, "ln_weight numel must equal hidden");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_swiglu_ln(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_proj.data_ptr<at::BFloat16>()),
        ln_weight.data_ptr<float>(),
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
    TORCH_CHECK(cos_cache.numel() == (int64_t)seq_len * half_dim, "cos_cache size mismatch");
    TORCH_CHECK(sin_cache.numel() == (int64_t)seq_len * half_dim, "sin_cache size mismatch");
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
    TORCH_CHECK(cos_cache.numel() == (int64_t)seq_len * half_dim, "cos_cache numel mismatch");
    TORCH_CHECK(sin_cache.numel() == (int64_t)seq_len * half_dim, "sin_cache numel mismatch");

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

void pcie_gradient_pack_py(at::Tensor bucket,
                            std::vector<std::tuple<at::Tensor, int64_t, int64_t>> chunks_in,
                            int sm_version)
{
    check_bf16(bucket, "bucket");
    TORCH_CHECK(!chunks_in.empty(), "chunks must not be empty");

    std::vector<PcieGradChunk> chunks;
    chunks.reserve(chunks_in.size());
    size_t total_elems = 0;
    for (size_t i = 0; i < chunks_in.size(); i++) {
        at::Tensor& t   = std::get<0>(chunks_in[i]);
        int64_t offset  = std::get<1>(chunks_in[i]);
        int64_t length  = std::get<2>(chunks_in[i]);
        check_bf16(t, ("chunks[" + std::to_string(i) + "].tensor").c_str());
        TORCH_CHECK(offset >= 0,    "chunk offset must be >= 0");
        TORCH_CHECK(length > 0,     "chunk length must be > 0");
        TORCH_CHECK(length % 8 == 0, "chunk length must be divisible by 8, got ", length);
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

// New: probe PCIe bandwidth
float probe_pcie_bandwidth_py(int src_device, int dst_device)
{
    TORCH_CHECK(src_device >= 0 && src_device < 16, "src_device out of range");
    TORCH_CHECK(dst_device >= 0 && dst_device < 16, "dst_device out of range");
    TORCH_CHECK(src_device != dst_device, "src_device must differ from dst_device");
    return probe_pcie_bandwidth(src_device, dst_device);
}

// New: adaptive chunk size
int64_t compute_adaptive_chunk_size_py(float pcie_bw_gbps)
{
    TORCH_CHECK(pcie_bw_gbps > 0.f, "pcie_bw_gbps must be > 0");
    return (int64_t)compute_adaptive_chunk_size(pcie_bw_gbps);
}

// New: per-SM bucket size
int64_t hetero_bucket_size_py(int sm_version)
{
    return (int64_t)hetero_bucket_size_elems(sm_version);
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
    check_int8(output, "output");
    check_fp32(scales,  "scales");
    check_bf16(input,   "input");
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
    check_int8(input,  "input");
    check_fp32(scales, "scales");
    TORCH_CHECK(output.numel() == input.numel(), "output/input numel mismatch");

    const size_t n_elems = (size_t)input.numel();
    const size_t n_tiles = (n_elems + 127) / 128;
    TORCH_CHECK((size_t)scales.numel() >= n_tiles, "scales buffer too small");

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

    TORCH_CHECK(hidden % 8 == 0, "hidden must be divisible by 8, got ", hidden);
    TORCH_CHECK(residual.sizes() == output.sizes(), "residual/output shape mismatch");
    TORCH_CHECK(input.sizes()    == output.sizes(), "input/output shape mismatch");
    TORCH_CHECK(ln_weight.numel() == hidden, "ln_weight numel must equal hidden");

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
    at::Tensor local_logit   = at::zeros({batch}, opts);

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
    TORCH_CHECK(global_max.numel() == global_sum_exp.numel(), "shape mismatch");
    TORCH_CHECK(global_max.numel() == global_logit.numel(),   "shape mismatch");

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
    check_bf16(d_logits,    "d_logits");
    check_bf16(logits,      "logits");
    check_fp32(global_max,  "global_max");
    check_fp32(log_sum_exp, "log_sum_exp");
    TORCH_CHECK(labels.scalar_type() == at::ScalarType::Int, "labels must be Int32");
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
// fused_gradient_allreduce bindings (INT8 compressed allreduce)
// ---------------------------------------------------------------------------

void gradient_compress_py(at::Tensor out_int8,
                            at::Tensor out_scale,
                            at::Tensor input,
                            int sm_version)
{
    check_int8(out_int8,  "out_int8");
    check_fp32(out_scale, "out_scale");
    check_bf16(input,     "input");
    const size_t n_elems = (size_t)input.numel();
    TORCH_CHECK((size_t)out_int8.numel() >= n_elems, "out_int8 too small");
    const size_t n_scale = (n_elems + 255) / 256;
    TORCH_CHECK((size_t)out_scale.numel() >= n_scale, "out_scale too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gradient_compress(
        reinterpret_cast<int8_t*>(out_int8.data_ptr<int8_t>()),
        out_scale.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        n_elems, sm_version, stream);
}

void gradient_decompress_py(at::Tensor output,
                              at::Tensor int8_data,
                              at::Tensor scale_buf,
                              int sm_version)
{
    check_bf16(output,   "output");
    check_int8(int8_data, "int8_data");
    check_fp32(scale_buf, "scale_buf");
    const size_t n_elems = (size_t)output.numel();
    TORCH_CHECK((size_t)int8_data.numel() >= n_elems, "int8_data too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gradient_decompress(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const int8_t*>(int8_data.data_ptr<int8_t>()),
        scale_buf.data_ptr<float>(),
        n_elems, sm_version, stream);
}

void int8_ring_reduce_step_py(at::Tensor dst_int8,
                               at::Tensor dst_scale,
                               at::Tensor src_int8,
                               at::Tensor src_scale,
                               int sm_version)
{
    check_int8(dst_int8,  "dst_int8");
    check_fp32(dst_scale, "dst_scale");
    check_int8(src_int8,  "src_int8");
    check_fp32(src_scale, "src_scale");
    TORCH_CHECK(dst_int8.numel() == src_int8.numel(), "int8 size mismatch");

    const size_t n_elems = (size_t)dst_int8.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_int8_ring_reduce_step(
        reinterpret_cast<int8_t*>(dst_int8.data_ptr<int8_t>()),
        dst_scale.data_ptr<float>(),
        reinterpret_cast<const int8_t*>(src_int8.data_ptr<int8_t>()),
        src_scale.data_ptr<float>(),
        n_elems, sm_version, stream);
}

void gradient_allreduce_finalise_py(at::Tensor scale_buf,
                                     int64_t n_elems,
                                     int world_size)
{
    check_fp32(scale_buf, "scale_buf");
    TORCH_CHECK(n_elems > 0,    "n_elems must be > 0");
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
// PYBIND11_MODULE
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "DeepSpeed hetero_reduce: fused BF16 reduce-scatter + SwiGLU-LN + "
              "RoPE + PCIe allreduce + tier activation offload + INT8 gradient "
              "allreduce kernels for heterogeneous GPU clusters (SM 8.6 / 9.0 / 12.0).";

    // ── Core reduce-scatter ──────────────────────────────────────────────
    m.def("fused_bf16_reduce",      &fused_bf16_reduce_py,
          py::arg("output"), py::arg("inputs"), py::arg("sm_version") = 86);

    m.def("hetero_reduce_scatter",  &hetero_reduce_scatter_py,
          py::arg("output"), py::arg("inputs"), py::arg("shard_offset"),
          py::arg("shard_count"), py::arg("sm_version") = 86);

    m.def("compute_shard_ranges",   &compute_shard_ranges_py,
          py::arg("sm_versions"), py::arg("total_elems"));

    // ── Fused activations ────────────────────────────────────────────────
    m.def("fused_swiglu_ln",        &fused_swiglu_ln_py,
          py::arg("output"), py::arg("gate_proj"), py::arg("up_proj"),
          py::arg("ln_weight"), py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);

    // ── RoPE ─────────────────────────────────────────────────────────────
    m.def("rope_cache",             &rope_cache_py,
          py::arg("cos_cache"), py::arg("sin_cache"),
          py::arg("seq_len"), py::arg("head_dim"),
          py::arg("base") = 10000.f, py::arg("pos_offset") = 0);

    m.def("fused_rope_hetero",      &fused_rope_hetero_py,
          py::arg("output"), py::arg("input"),
          py::arg("cos_cache"), py::arg("sin_cache"),
          py::arg("neox_style") = true, py::arg("sm_version") = 86);

    // ── PCIe allreduce ───────────────────────────────────────────────────
    m.def("pcie_gradient_pack",     &pcie_gradient_pack_py,
          py::arg("bucket"), py::arg("chunks"), py::arg("sm_version") = 86);

    m.def("pcie_ring_reduce",       &pcie_ring_reduce_py,
          py::arg("dst"), py::arg("src"), py::arg("sm_version") = 86);

    m.def("pcie_allreduce_finalise", &pcie_allreduce_finalise_py,
          py::arg("out"), py::arg("src"),
          py::arg("world_size"), py::arg("sm_version") = 86);

    m.def("pcie_bucket_size",        &pcie_bucket_size_py,
          py::arg("pcie_bw_gbps") = 32.f);

    m.def("probe_pcie_bandwidth",    &probe_pcie_bandwidth_py,
          "Measure PCIe bandwidth between two CUDA devices in GB/s.\n"
          "Args:\n"
          "  src_device (int): CUDA device ordinal of sender\n"
          "  dst_device (int): CUDA device ordinal of receiver\n"
          "Returns: float (measured GB/s)",
          py::arg("src_device"), py::arg("dst_device"));

    m.def("compute_adaptive_chunk_size", &compute_adaptive_chunk_size_py,
          "Compute ring-allreduce chunk size targeting ~5 ms of PCIe overlap.\n"
          "Args:\n"
          "  pcie_bw_gbps (float): PCIe bandwidth in GB/s\n"
          "Returns: int (chunk size in bytes, 16-byte aligned)",
          py::arg("pcie_bw_gbps"));

    m.def("hetero_bucket_size_elems", &hetero_bucket_size_py,
          "Per-SM recommended gradient bucket size in BF16 elements.\n"
          "Args:\n"
          "  sm_version (int): 86, 90, or 120\n"
          "Returns: int (elements)",
          py::arg("sm_version") = 86);

    // ── Activation offload ───────────────────────────────────────────────
    m.def("activation_pack",        &activation_pack_py,
          py::arg("output"), py::arg("inputs"), py::arg("sm_version") = 86);

    m.def("activation_unpack",      &activation_unpack_py,
          py::arg("outputs"), py::arg("flat"), py::arg("sm_version") = 86);

    m.def("quantise_bf16_to_int8",   &quantise_bf16_to_int8_py,
          py::arg("output"), py::arg("scales"), py::arg("input"));

    m.def("dequantise_int8_to_bf16", &dequantise_int8_to_bf16_py,
          py::arg("output"), py::arg("input"), py::arg("scales"));

    m.def("compute_offload_budget",  &compute_offload_budget_py,
          py::arg("total_act_bytes"), py::arg("vram_free_bytes"),
          py::arg("headroom_frac") = 0.1f);

    // ── LayerNorm + residual ─────────────────────────────────────────────
    m.def("fused_layernorm_residual", &fused_layernorm_residual_py,
          py::arg("output"), py::arg("residual"), py::arg("input"),
          py::arg("ln_weight"), py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);

    // ── Cross-entropy TP ─────────────────────────────────────────────────
    m.def("cross_entropy_tp_forward",   &cross_entropy_tp_forward_py,
          py::arg("logits"), py::arg("labels"),
          py::arg("shard_offset") = 0, py::arg("sm_version") = 86);

    m.def("cross_entropy_tp_loss",      &cross_entropy_tp_loss_py,
          py::arg("global_max"), py::arg("global_sum_exp"), py::arg("global_logit"));

    m.def("cross_entropy_tp_backward",  &cross_entropy_tp_backward_py,
          py::arg("d_logits"), py::arg("logits"), py::arg("labels"),
          py::arg("global_max"), py::arg("log_sum_exp"),
          py::arg("shard_offset") = 0, py::arg("inv_batch") = 1.f,
          py::arg("sm_version") = 86);

    // ── INT8 compressed gradient allreduce ───────────────────────────────
    m.def("gradient_compress",       &gradient_compress_py,
          "Compress BF16 gradient to INT8 + per-block FP32 scale.\n"
          "Args:\n"
          "  out_int8  (Tensor Int8  [n_elems]): compressed output\n"
          "  out_scale (Tensor FP32  [n_blocks]): per-block scales\n"
          "  input     (Tensor BF16  [n_elems]): gradient input\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("out_int8"), py::arg("out_scale"),
          py::arg("input"), py::arg("sm_version") = 86);

    m.def("gradient_decompress",     &gradient_decompress_py,
          "Decompress INT8 + per-block FP32 scale back to BF16.\n"
          "Args:\n"
          "  output    (Tensor BF16  [n_elems]): decompressed output\n"
          "  int8_data (Tensor Int8  [n_elems]): compressed input\n"
          "  scale_buf (Tensor FP32  [n_blocks]): per-block scales\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("output"), py::arg("int8_data"),
          py::arg("scale_buf"), py::arg("sm_version") = 86);

    m.def("int8_ring_reduce_step",   &int8_ring_reduce_step_py,
          "Fused INT8 ring-allreduce step: dequant + add + requant.\n"
          "Args:\n"
          "  dst_int8  (Tensor Int8  [n_elems]): accumulator (in/out)\n"
          "  dst_scale (Tensor FP32  [n_blocks]): accumulator scales (in/out)\n"
          "  src_int8  (Tensor Int8  [n_elems]): received peer data\n"
          "  src_scale (Tensor FP32  [n_blocks]): received peer scales\n"
          "  sm_version (int): 86, 90, or 120",
          py::arg("dst_int8"), py::arg("dst_scale"),
          py::arg("src_int8"), py::arg("src_scale"),
          py::arg("sm_version") = 86);

    m.def("gradient_allreduce_finalise", &gradient_allreduce_finalise_py,
          "Divide per-block scales by world_size after ring allreduce.\n"
          "Args:\n"
          "  scale_buf  (Tensor FP32): per-block scales (in/out)\n"
          "  n_elems    (int): total gradient elements\n"
          "  world_size (int): number of participating GPUs",
          py::arg("scale_buf"), py::arg("n_elems"), py::arg("world_size"));

    m.def("gradient_compress_bytes", &gradient_compress_bytes_py,
          "INT8 staging buffer size in bytes for n_elems gradient elements.",
          py::arg("n_elems"));

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

    m.def("gradient_scale_bytes",    &gradient_scale_bytes_py,
          "Per-block scale buffer size in bytes for n_elems gradient elements.",
          py::arg("n_elems"));

}
