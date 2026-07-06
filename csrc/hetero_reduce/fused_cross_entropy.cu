// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_cross_entropy.cu
 *
 * Heterogeneous-GPU fused cross-entropy loss kernel.
 *
 * Design
 * ------
 * In a heterogeneous cluster each GPU holds a *partition* of the vocabulary
 * (vocab-parallel / tensor-parallel along the vocab dimension).  Computing
 * cross-entropy requires a global log-softmax, which in turn needs:
 *
 *   1.  Global max of logits across all vocab partitions  (for numerical
 *       stability of the softmax).
 *   2.  Global sum of exp(logit - global_max) across all partitions (the
 *       softmax denominator).
 *   3.  Gathering the logit at the target position (which lives on exactly
 *       one GPU) and computing the final scalar loss.
 *
 * This file provides efficient per-partition CUDA kernels for steps (1)–(3).
 * The cross-partition reductions (global max, global exp-sum) are performed
 * via the hetero_reduce primitives already present in this library.
 *
 * Kernel map
 * ----------
 *   fused_local_max_and_expsum_kernel
 *       Phase 1+2 fused:  For each row, computes the local max of the
 *       partition logits, then in a second pass (or online via the
 *       shifted-exp trick) computes sum(exp(x - local_max)).  Outputs
 *       per-row (local_max, local_expsum) pairs.
 *
 *   adjust_expsum_kernel
 *       After the global max is known (via allreduce of local maxes),
 *       corrects each local exp-sum:
 *           corrected_i = local_expsum_i * exp(local_max_i - global_max)
 *       The sum of corrected exp-sums across partitions equals the global
 *       softmax denominator.
 *
 *   cross_entropy_loss_kernel
 *       Phase 3:  Given global_max, global_expsum, the logit at the target
 *       position, and the target index, computes:
 *           loss = log(global_expsum) - (target_logit - global_max)
 *       Optionally writes per-token loss or reduces to a scalar mean.
 *
 *   gather_target_logit_kernel
 *       Extracts the logit value at the target position from the local
 *       partition if the target falls within [vocab_start, vocab_end).
 *
 *   fused_softmax_grad_kernel
 *       Backward pass:  Given the upstream grad (scalar per row), global_max,
 *       global_expsum, and the local logit partition, computes the gradient
 *       w.r.t. the local logits:
 *           d_logit_i = (exp(logit_i - global_max) / global_expsum - 1_{i==target}) * grad
 *
 * All kernels are templated on SM version (86 / 90 / 120) with policy-based
 * block sizing identical to the rest of the hetero_reduce library.
 *
 * Reference: Megatron-LM fused cross-entropy (vocab-parallel softmax).
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "hetero_reduce.h"

// ---------------------------------------------------------------------------
// Kernel policy — mirrors the rest of hetero_reduce
// ---------------------------------------------------------------------------

template <int SmVersion>
struct CrossEntropyPolicy;

template <>
struct CrossEntropyPolicy<86> {   // A6000 / Ada
    static constexpr int kBlockSize     = 256;
    static constexpr int kRowsPerBlock  = 1;
    static constexpr int kVecWidth      = 4;   // BF16x4 loads (8 bytes)
};

template <>
struct CrossEntropyPolicy<90> {   // H100
    static constexpr int kBlockSize     = 512;
    static constexpr int kRowsPerBlock  = 1;
    static constexpr int kVecWidth      = 8;   // BF16x8 loads (16 bytes)
};

template <>
struct CrossEntropyPolicy<120> {  // Blackwell
    static constexpr int kBlockSize     = 512;
    static constexpr int kRowsPerBlock  = 1;
    static constexpr int kVecWidth      = 8;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v)
{
    return __float2bfloat16(v);
}

// Warp-level reduction helpers
__device__ __forceinline__ float warp_reduce_max(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reductions using shared memory
template <int BlockSize>
__device__ float block_reduce_max(float val, float* smem)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    constexpr int kNumWarps = BlockSize / 32;

    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lane < kNumWarps) ? smem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    __syncthreads();
    return val;
}

template <int BlockSize>
__device__ float block_reduce_sum(float val, float* smem)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    constexpr int kNumWarps = BlockSize / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lane < kNumWarps) ? smem[lane] : 0.f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}

// ---------------------------------------------------------------------------
// Kernel 1: fused_local_max_and_expsum_kernel
//
// For each row (token), compute the local max and sum(exp(x - local_max))
// over the local vocab partition.  Uses online numerically-stable softmax
// (the "log-sum-exp trick").
//
// Input:  logits [batch_size, local_vocab_size]  (BF16)
// Output: local_max    [batch_size]              (FP32)
//         local_expsum [batch_size]              (FP32)
// ---------------------------------------------------------------------------

template <int SmVersion>
__global__ void fused_local_max_and_expsum_kernel(
    const __nv_bfloat16* __restrict__ logits,   // [B, V_local]
    float* __restrict__               local_max,     // [B]
    float* __restrict__               local_expsum,  // [B]
    int                               batch_size,
    int                               local_vocab_size)
{
    using Policy = CrossEntropyPolicy<SmVersion>;
    constexpr int kBlock = Policy::kBlockSize;

    extern __shared__ float smem[];  // kBlock/32 floats

    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_logits = logits + (size_t)row * local_vocab_size;

    // ---- Pass 1: find local max ----
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < local_vocab_size; i += kBlock) {
        float v = bf16_to_float(row_logits[i]);
        thread_max = fmaxf(thread_max, v);
    }
    float row_max = block_reduce_max<kBlock>(thread_max, smem);

    // Broadcast via smem
    if (threadIdx.x == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];
    __syncthreads();

    // ---- Pass 2: compute exp-sum using the stable shift ----
    float thread_sum = 0.f;
    for (int i = threadIdx.x; i < local_vocab_size; i += kBlock) {
        float v = bf16_to_float(row_logits[i]);
        thread_sum += expf(v - row_max);
    }
    float row_sum = block_reduce_sum<kBlock>(thread_sum, smem);

    if (threadIdx.x == 0) {
        local_max[row]    = row_max;
        local_expsum[row] = row_sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: adjust_expsum_kernel
//
// After an allreduce to find the global_max across all partitions, each
// partition corrects its local exp-sum:
//
//   corrected = local_expsum * exp(local_max - global_max)
//
// A subsequent allreduce of corrected values gives the global exp-sum.
//
// In-place: local_expsum is overwritten with the corrected value.
// ---------------------------------------------------------------------------

__global__ void adjust_expsum_kernel(
    float* __restrict__       local_expsum,   // [B], in-place
    const float* __restrict__ local_max,      // [B]
    const float* __restrict__ global_max,     // [B]
    int                       batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float diff = local_max[idx] - global_max[idx];
    // Guard against very large negative diff causing underflow
    float correction = (diff < -80.f) ? 0.f : expf(diff);
    local_expsum[idx] *= correction;
}

// ---------------------------------------------------------------------------
// Kernel 3: gather_target_logit_kernel
//
// Each GPU holds logits for vocab indices [vocab_start, vocab_start + V_local).
// The target index for each token may or may not fall in this range.
// If it does, we extract the logit value; otherwise we write 0.
// A subsequent allreduce (sum) across partitions yields the true target logit.
//
// We also output a mask indicating whether this partition owns the target,
// needed for the backward pass.
// ---------------------------------------------------------------------------

__global__ void gather_target_logit_kernel(
    const __nv_bfloat16* __restrict__ logits,        // [B, V_local]
    const int64_t* __restrict__       targets,       // [B]
    float* __restrict__               target_logit,  // [B]
    int* __restrict__                 target_mask,   // [B]  1 if target on this partition
    int                               batch_size,
    int                               local_vocab_size,
    int                               vocab_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int64_t target = targets[idx];
    int64_t local_idx = target - (int64_t)vocab_start;

    if (local_idx >= 0 && local_idx < (int64_t)local_vocab_size) {
        target_logit[idx] = bf16_to_float(
            logits[(size_t)idx * local_vocab_size + local_idx]);
        target_mask[idx] = 1;
    } else {
        target_logit[idx] = 0.f;
        target_mask[idx]  = 0;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: cross_entropy_loss_kernel
//
// Computes the final per-token cross-entropy loss:
//
//   loss_i = log(global_expsum_i) - (target_logit_i - global_max_i)
//          = log(global_expsum_i) + global_max_i - target_logit_i
//
// Optionally computes mean over all tokens.
//
// Ignore index: tokens with target == ignore_index get loss = 0.
// ---------------------------------------------------------------------------

__global__ void cross_entropy_loss_kernel(
    float* __restrict__         loss,            // [B]  per-token loss
    float* __restrict__         mean_loss,       // [1]  optional scalar mean (may be NULL)
    const float* __restrict__   global_max,      // [B]
    const float* __restrict__   global_expsum,   // [B]
    const float* __restrict__   target_logit,    // [B]
    const int64_t* __restrict__ targets,         // [B]
    int                         batch_size,
    int                         ignore_index)
{
    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float my_loss = 0.f;
    bool valid = false;

    if (idx < batch_size) {
        if ((int)targets[idx] != ignore_index) {
            float log_denominator = logf(global_expsum[idx]);
            float numerator_logit = target_logit[idx] - global_max[idx];
            my_loss = log_denominator - numerator_logit;
            valid = true;
        }
        loss[idx] = my_loss;
    }

    // Optionally compute mean via block reduction + atomicAdd
    if (mean_loss != nullptr) {
        // Count valid tokens and total loss for mean
        float valid_count = valid ? 1.f : 0.f;

        // Simple block-wide reduction for both
        float block_loss  = 0.f;
        float block_count = 0.f;

        // Warp reduce
        float warp_loss  = warp_reduce_sum(my_loss);
        float warp_count = warp_reduce_sum(valid_count);

        int lane = threadIdx.x & 31;
        int wid  = threadIdx.x >> 5;

        if (lane == 0) {
            smem[wid]     = warp_loss;
            smem[wid + 16] = warp_count;
        }
        __syncthreads();

        int num_warps = blockDim.x / 32;
        if (wid == 0) {
            block_loss  = (lane < num_warps) ? smem[lane]      : 0.f;
            block_count = (lane < num_warps) ? smem[lane + 16] : 0.f;
            block_loss  = warp_reduce_sum(block_loss);
            block_count = warp_reduce_sum(block_count);
        }

        if (threadIdx.x == 0) {
            // mean_loss[0] accumulates total loss; mean_loss[1] accumulates count
            atomicAdd(&mean_loss[0], block_loss);
            atomicAdd(&mean_loss[1], block_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: fused_softmax_grad_kernel (backward pass)
//
// For each element in the local vocab partition:
//
//   softmax_i = exp(logit_i - global_max) / global_expsum
//   d_logit_i = (softmax_i - 1_{i == target}) * grad_output
//
// The grad_output is a per-row scalar (or per-token scalar for per-token loss).
// ---------------------------------------------------------------------------

template <int SmVersion>
__global__ void fused_softmax_grad_kernel(
    __nv_bfloat16* __restrict__       d_logits,       // [B, V_local]
    const __nv_bfloat16* __restrict__ logits,         // [B, V_local]
    const float* __restrict__         global_max,     // [B]
    const float* __restrict__         global_expsum,  // [B]
    const float* __restrict__         grad_output,    // [B] per-token upstream grad
    const int64_t* __restrict__       targets,        // [B]
    int                               batch_size,
    int                               local_vocab_size,
    int                               vocab_start,
    int                               ignore_index)
{
    using Policy = CrossEntropyPolicy<SmVersion>;
    constexpr int kBlock = Policy::kBlockSize;

    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int64_t target = targets[row];
    const float g_max    = global_max[row];
    const float g_expsum = global_expsum[row];
    const float grad     = grad_output[row];

    // Ignore-index: zero gradient
    const bool is_ignored = ((int)target == ignore_index);

    const __nv_bfloat16* row_logits = logits   + (size_t)row * local_vocab_size;
    __nv_bfloat16*       row_grad   = d_logits + (size_t)row * local_vocab_size;

    const int64_t local_target = target - (int64_t)vocab_start;
    const float inv_expsum = 1.f / g_expsum;

    for (int i = threadIdx.x; i < local_vocab_size; i += kBlock) {
        if (is_ignored) {
            row_grad[i] = float_to_bf16(0.f);
        } else {
            float logit_val = bf16_to_float(row_logits[i]);
            float softmax_val = expf(logit_val - g_max) * inv_expsum;

            // Subtract 1 if this is the target
            float indicator = (i == (int)local_target) ? 1.f : 0.f;
            float d_val = (softmax_val - indicator) * grad;

            row_grad[i] = float_to_bf16(d_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: finalize_mean_loss_kernel
//
// Divides accumulated loss by accumulated count to produce final mean.
// ---------------------------------------------------------------------------

__global__ void finalize_mean_loss_kernel(float* __restrict__ mean_loss)
{
    // mean_loss[0] = total_loss, mean_loss[1] = total_count
    if (threadIdx.x == 0) {
        float count = mean_loss[1];
        mean_loss[0] = (count > 0.f) ? (mean_loss[0] / count) : 0.f;
    }
}

// ===========================================================================
// Launch wrappers
// ===========================================================================

void launch_fused_local_max_expsum(
    const __nv_bfloat16* logits,
    float*               local_max,
    float*               local_expsum,
    int                  batch_size,
    int                  local_vocab_size,
    int                  sm_version,
    cudaStream_t         stream)
{
    auto launch = [&](auto SmTag) {
        constexpr int SmV = decltype(SmTag)::value;
        using Policy = CrossEntropyPolicy<SmV>;
        constexpr int kBlock = Policy::kBlockSize;
        int smem_bytes = (kBlock / 32) * sizeof(float);
        fused_local_max_and_expsum_kernel<SmV>
            <<<batch_size, kBlock, smem_bytes, stream>>>(
                logits, local_max, local_expsum, batch_size, local_vocab_size);
    };

    if (sm_version >= 120)
        launch(std::integral_constant<int, 120>{});
    else if (sm_version >= 90)
        launch(std::integral_constant<int, 90>{});
    else
        launch(std::integral_constant<int, 86>{});
}

void launch_adjust_expsum(
    float*       local_expsum,
    const float* local_max,
    const float* global_max,
    int          batch_size,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    int grid = (batch_size + kBlock - 1) / kBlock;
    adjust_expsum_kernel<<<grid, kBlock, 0, stream>>>(
        local_expsum, local_max, global_max, batch_size);
}

void launch_gather_target_logit(
    const __nv_bfloat16* logits,
    const int64_t*       targets,
    float*               target_logit,
    int*                 target_mask,
    int                  batch_size,
    int                  local_vocab_size,
    int                  vocab_start,
    cudaStream_t         stream)
{
    constexpr int kBlock = 256;
    int grid = (batch_size + kBlock - 1) / kBlock;
    gather_target_logit_kernel<<<grid, kBlock, 0, stream>>>(
        logits, targets, target_logit, target_mask,
        batch_size, local_vocab_size, vocab_start);
}

void launch_cross_entropy_loss(
    float*         loss,
    float*         mean_loss,
    const float*   global_max,
    const float*   global_expsum,
    const float*   target_logit,
    const int64_t* targets,
    int            batch_size,
    int            ignore_index,
    cudaStream_t   stream)
{
    constexpr int kBlock = 256;
    int grid = (batch_size + kBlock - 1) / kBlock;
    int smem_bytes = 32 * sizeof(float);  // 16 for loss + 16 for count
    cross_entropy_loss_kernel<<<grid, kBlock, smem_bytes, stream>>>(
        loss, mean_loss, global_max, global_expsum, target_logit,
        targets, batch_size, ignore_index);

    // Finalize mean
    if (mean_loss != nullptr) {
        finalize_mean_loss_kernel<<<1, 1, 0, stream>>>(mean_loss);
    }
}

void launch_fused_cross_entropy_backward(
    __nv_bfloat16*       d_logits,
    const __nv_bfloat16* logits,
    const float*         global_max,
    const float*         global_expsum,
    const float*         grad_output,
    const int64_t*       targets,
    int                  batch_size,
    int                  local_vocab_size,
    int                  vocab_start,
    int                  ignore_index,
    int                  sm_version,
    cudaStream_t         stream)
{
    auto launch = [&](auto SmTag) {
        constexpr int SmV = decltype(SmTag)::value;
        using Policy = CrossEntropyPolicy<SmV>;
        constexpr int kBlock = Policy::kBlockSize;
        fused_softmax_grad_kernel<SmV>
            <<<batch_size, kBlock, 0, stream>>>(
                d_logits, logits, global_max, global_expsum,
                grad_output, targets,
                batch_size, local_vocab_size, vocab_start, ignore_index);
    };

    if (sm_version >= 120)
        launch(std::integral_constant<int, 120>{});
    else if (sm_version >= 90)
        launch(std::integral_constant<int, 90>{});
    else
        launch(std::integral_constant<int, 86>{});
}
