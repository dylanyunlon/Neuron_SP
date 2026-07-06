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

// ---------------------------------------------------------------------------
// fused_cross_entropy bindings
// ---------------------------------------------------------------------------

void fused_local_max_expsum_py(at::Tensor logits,
                                at::Tensor local_max,
                                at::Tensor local_expsum,
                                int sm_version)
{
    check_bf16(logits, "logits");
    check_fp32(local_max, "local_max");
    check_fp32(local_expsum, "local_expsum");

    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [B, V_local]");
    const int batch = static_cast<int>(logits.size(0));
    const int local_vocab = static_cast<int>(logits.size(1));
    TORCH_CHECK(local_max.numel() >= batch, "local_max too small");
    TORCH_CHECK(local_expsum.numel() >= batch, "local_expsum too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_local_max_expsum(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        local_max.data_ptr<float>(),
        local_expsum.data_ptr<float>(),
        batch, local_vocab, sm_version, stream);
}

void adjust_expsum_py(at::Tensor local_expsum,
                       at::Tensor local_max,
                       at::Tensor global_max,
                       int batch_size)
{
    check_fp32(local_expsum, "local_expsum");
    check_fp32(local_max, "local_max");
    check_fp32(global_max, "global_max");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_adjust_expsum(
        local_expsum.data_ptr<float>(),
        local_max.data_ptr<float>(),
        global_max.data_ptr<float>(),
        batch_size, stream);
}

std::tuple<at::Tensor, at::Tensor>
gather_target_logit_py(at::Tensor logits,
                        at::Tensor targets,
                        int vocab_start)
{
    check_bf16(logits, "logits");
    TORCH_CHECK(targets.scalar_type() == at::ScalarType::Long,
                "targets must be Int64");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous());
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [B, V_local]");

    const int batch = static_cast<int>(logits.size(0));
    const int local_vocab = static_cast<int>(logits.size(1));

    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(logits.device());
    auto opts_i = at::TensorOptions().dtype(at::kInt).device(logits.device());
    at::Tensor target_logit = at::empty({batch}, opts_f);
    at::Tensor target_mask  = at::empty({batch}, opts_i);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gather_target_logit(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        targets.data_ptr<int64_t>(),
        target_logit.data_ptr<float>(),
        target_mask.data_ptr<int>(),
        batch, local_vocab, vocab_start, stream);

    return std::make_tuple(target_logit, target_mask);
}

at::Tensor cross_entropy_loss_py(at::Tensor global_max,
                                   at::Tensor global_expsum,
                                   at::Tensor target_logit,
                                   at::Tensor targets,
                                   bool compute_mean,
                                   int ignore_index)
{
    check_fp32(global_max, "global_max");
    check_fp32(global_expsum, "global_expsum");
    check_fp32(target_logit, "target_logit");
    TORCH_CHECK(targets.scalar_type() == at::ScalarType::Long,
                "targets must be Int64");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous());

    const int batch = static_cast<int>(global_max.numel());
    auto opts = at::TensorOptions().dtype(at::kFloat).device(global_max.device());
    at::Tensor loss = at::empty({batch}, opts);

    float* mean_ptr = nullptr;
    at::Tensor mean_loss;
    if (compute_mean) {
        mean_loss = at::zeros({2}, opts);  // [total_loss, total_count]
        mean_ptr = mean_loss.data_ptr<float>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cross_entropy_loss(
        loss.data_ptr<float>(),
        mean_ptr,
        global_max.data_ptr<float>(),
        global_expsum.data_ptr<float>(),
        target_logit.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        batch, ignore_index, stream);

    if (compute_mean) {
        return mean_loss.slice(0, 0, 1);  // return just the mean scalar
    }
    return loss;
}

void fused_cross_entropy_backward_py(at::Tensor d_logits,
                                       at::Tensor logits,
                                       at::Tensor global_max,
                                       at::Tensor global_expsum,
                                       at::Tensor grad_output,
                                       at::Tensor targets,
                                       int vocab_start,
                                       int ignore_index,
                                       int sm_version)
{
    check_bf16(d_logits, "d_logits");
    check_bf16(logits, "logits");
    check_fp32(global_max, "global_max");
    check_fp32(global_expsum, "global_expsum");
    check_fp32(grad_output, "grad_output");
    TORCH_CHECK(targets.scalar_type() == at::ScalarType::Long,
                "targets must be Int64");

    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [B, V_local]");
    TORCH_CHECK(d_logits.sizes() == logits.sizes(), "d_logits/logits shape mismatch");

    const int batch = static_cast<int>(logits.size(0));
    const int local_vocab = static_cast<int>(logits.size(1));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_cross_entropy_backward(
        reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        global_max.data_ptr<float>(),
        global_expsum.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        batch, local_vocab, vocab_start, ignore_index, sm_version, stream);
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
// PYBIND11_MODULE
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "DeepSpeed hetero_reduce: fused BF16 reduce-scatter + SwiGLU-LN + "
              "RoPE + PCIe allreduce + tier activation offload + "
              "vocab-parallel cross-entropy kernels "
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
    // fused_cross_entropy
    // -----------------------------------------------------------------------
    m.def("fused_local_max_expsum",
          &fused_local_max_expsum_py,
          "Compute per-row local max and exp-sum over a vocab partition.\n"
          "Phase 1+2 of heterogeneous vocab-parallel cross-entropy.\n"
          "Args:\n"
          "  logits       (Tensor BF16 [B, V_local]): local logit partition\n"
          "  local_max    (Tensor FP32 [B]): output per-row local max\n"
          "  local_expsum (Tensor FP32 [B]): output per-row exp-sum\n"
          "  sm_version   (int): 86, 90, or 120",
          py::arg("logits"),
          py::arg("local_max"),
          py::arg("local_expsum"),
          py::arg("sm_version") = 86);

    m.def("adjust_expsum",
          &adjust_expsum_py,
          "Correct local exp-sum after global max allreduce.\n"
          "In-place: local_expsum *= exp(local_max - global_max).\n"
          "Args:\n"
          "  local_expsum (Tensor FP32 [B]): corrected in-place\n"
          "  local_max    (Tensor FP32 [B]): per-row local max\n"
          "  global_max   (Tensor FP32 [B]): per-row global max\n"
          "  batch_size   (int): number of tokens",
          py::arg("local_expsum"),
          py::arg("local_max"),
          py::arg("global_max"),
          py::arg("batch_size"));

    m.def("gather_target_logit",
          &gather_target_logit_py,
          "Extract target logit from local partition.\n"
          "Returns (target_logit, target_mask) tensors.\n"
          "Args:\n"
          "  logits      (Tensor BF16 [B, V_local]): local logit partition\n"
          "  targets     (Tensor Int64 [B]): global target indices\n"
          "  vocab_start (int): starting global vocab index of this partition",
          py::arg("logits"),
          py::arg("targets"),
          py::arg("vocab_start") = 0);

    m.def("cross_entropy_loss",
          &cross_entropy_loss_py,
          "Compute final cross-entropy loss from global softmax statistics.\n"
          "loss = log(global_expsum) - (target_logit - global_max).\n"
          "Returns per-token loss or scalar mean.\n"
          "Args:\n"
          "  global_max    (Tensor FP32 [B]): global max across partitions\n"
          "  global_expsum (Tensor FP32 [B]): global exp-sum denominator\n"
          "  target_logit  (Tensor FP32 [B]): logit at target position\n"
          "  targets       (Tensor Int64 [B]): target indices\n"
          "  compute_mean  (bool): if True, return scalar mean loss\n"
          "  ignore_index  (int): target value to ignore (default -100)",
          py::arg("global_max"),
          py::arg("global_expsum"),
          py::arg("target_logit"),
          py::arg("targets"),
          py::arg("compute_mean") = true,
          py::arg("ignore_index") = -100);

    m.def("fused_cross_entropy_backward",
          &fused_cross_entropy_backward_py,
          "Backward pass for vocab-parallel cross-entropy.\n"
          "d_logit = (softmax - indicator) * grad_output.\n"
          "Args:\n"
          "  d_logits      (Tensor BF16 [B, V_local]): output gradient\n"
          "  logits        (Tensor BF16 [B, V_local]): forward logits\n"
          "  global_max    (Tensor FP32 [B]): global max\n"
          "  global_expsum (Tensor FP32 [B]): global exp-sum\n"
          "  grad_output   (Tensor FP32 [B]): upstream per-token gradient\n"
          "  targets       (Tensor Int64 [B]): target indices\n"
          "  vocab_start   (int): starting global vocab index\n"
          "  ignore_index  (int): target value to ignore\n"
          "  sm_version    (int): 86, 90, or 120",
          py::arg("d_logits"),
          py::arg("logits"),
          py::arg("global_max"),
          py::arg("global_expsum"),
          py::arg("grad_output"),
          py::arg("targets"),
          py::arg("vocab_start") = 0,
          py::arg("ignore_index") = -100,
          py::arg("sm_version") = 86);

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
}
