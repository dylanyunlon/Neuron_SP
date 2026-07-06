// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_cross_entropy.cu
 *
 * Heterogeneous-vocab-partition fused cross-entropy loss kernel.
 *
 * In a heterogeneous GPU cluster the vocabulary is sharded across tiers so
 * that each GPU computes logits only for its local vocab partition.  The
 * standard CE loss   L = -log(softmax(z)_y)   requires the global log-sum-exp
 * across ALL vocab entries.  This kernel computes:
 *
 *   1. Per-partition local max and local log-sum-exp (forward kernel).
 *   2. Accepts the *global* log-sum-exp computed after an allreduce of the
 *      per-partition statistics, and produces per-token losses + dlogits
 *      (backward/loss-combine kernel).
 *
 * BF16 logits, FP32 accumulation, BF16 gradient output.
 * SM-dispatched for SM 8.6 / 9.0 / 12.0 (via template + runtime switch).
 *
 * Shapes
 * ------
 *   logits        : [batch, local_vocab]  (BF16, the partition owned by this tier)
 *   targets       : [batch]               (int64, global vocab indices)
 *   vocab_offset  : int                   (start index of this tier's partition)
 *   local_vocab   : int                   (size of this tier's partition)
 *
 * Forward outputs (per-token, FP32):
 *   local_max     : [batch]     max over local vocab
 *   local_lse     : [batch]     log-sum-exp over local vocab (un-shifted by global max)
 *   local_target_logit : [batch] logit at the target position (0 if target not local)
 *   target_is_local    : [batch] 1.0 if target falls in this partition, else 0.0
 *
 * Loss-combine / backward outputs:
 *   losses        : [batch]               (FP32 scalar losses)
 *   dlogits       : [batch, local_vocab]  (BF16 gradients)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "hetero_reduce.h"

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int kWarpSize     = 32;
static constexpr int kBlockThreads = 256;   // threads per block
static constexpr int kVecWidth     = 8;     // BF16 elements per vectorised load

// ---------------------------------------------------------------------------
// Warp-level primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float val)
{
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduce via shared memory
// ---------------------------------------------------------------------------

__device__ float block_reduce_max(float val, float* smem)
{
    const int lane = threadIdx.x % kWarpSize;
    const int wid  = threadIdx.x / kWarpSize;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    const int num_warps = kBlockThreads / kWarpSize;
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    __syncthreads();
    return val;
}

__device__ float block_reduce_sum(float val, float* smem)
{
    const int lane = threadIdx.x % kWarpSize;
    const int wid  = threadIdx.x / kWarpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    const int num_warps = kBlockThreads / kWarpSize;
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    __syncthreads();
    return val;
}

// ---------------------------------------------------------------------------
// Kernel policy: tile sizes per SM generation
// ---------------------------------------------------------------------------

template <int SmVer>
struct CEPolicy {
    static constexpr int kTileV = 1024;   // vocab elements per tile
};

template <>
struct CEPolicy<90> {
    static constexpr int kTileV = 2048;   // H100: larger L2
};

template <>
struct CEPolicy<120> {
    static constexpr int kTileV = 1536;   // Blackwell: moderate
};

// ---------------------------------------------------------------------------
// Forward kernel: compute local max, local log-sum-exp, target logit
// ---------------------------------------------------------------------------
// One block per batch row.  Iterates over the local vocab in tiles.

template <int SmVer>
__global__ void __launch_bounds__(kBlockThreads)
fused_ce_forward_kernel(const __nv_bfloat16* __restrict__ logits,   // [B, V_local]
                        const int64_t*       __restrict__ targets,  // [B]
                        float*               __restrict__ local_max,
                        float*               __restrict__ local_lse,
                        float*               __restrict__ local_target_logit,
                        float*               __restrict__ target_is_local,
                        int                               local_vocab,
                        int                               vocab_offset)
{
    const int row = blockIdx.x;  // one block per batch element

    const __nv_bfloat16* row_logits = logits + (size_t)row * local_vocab;
    const int64_t global_target     = targets[row];

    // Determine if this partition owns the target token.
    const bool owns_target = (global_target >= vocab_offset) &&
                             (global_target <  vocab_offset + local_vocab);
    const int  local_target_idx = owns_target ? (int)(global_target - vocab_offset) : -1;

    __shared__ float smem[kBlockThreads / kWarpSize];

    // --- Pass 1: find local max ---
    float thread_max = -INFINITY;
    float target_val = 0.0f;

    for (int v = threadIdx.x; v < local_vocab; v += kBlockThreads) {
        float z = __bfloat162float(row_logits[v]);
        thread_max = fmaxf(thread_max, z);
        if (v == local_target_idx) target_val = z;
    }

    float row_max = block_reduce_max(thread_max, smem);

    // Broadcast target_val across block (only one thread has it).
    if (owns_target) {
        // Reduce-sum works because only one thread has nonzero target_val
        // if local_target_idx is unique — but safer to use atomicExch in smem.
        // We use a small shared buffer for this.
    }
    __shared__ float s_target_val;
    if (threadIdx.x == 0) s_target_val = 0.0f;
    __syncthreads();
    if (owns_target && (threadIdx.x * 1 <= local_target_idx) &&
        (local_target_idx < (int)((threadIdx.x + 1)))) {
        // This won't work — threadIdx stride != 1. Use atomicExch.
    }
    // Simpler approach: the thread that handled local_target_idx writes it.
    if (local_target_idx >= 0 && (local_target_idx % kBlockThreads == (int)threadIdx.x)) {
        // This thread visited local_target_idx in the loop above (since
        // loop stride is kBlockThreads and v starts at threadIdx.x).
        // Actually the loop visits v = threadIdx.x, threadIdx.x + kBlockThreads, ...
        // So the thread that covers local_target_idx is (local_target_idx % kBlockThreads).
    }
    // Let's just re-read it in a clean way for the owning thread:
    if (owns_target && (local_target_idx % kBlockThreads == (int)threadIdx.x)) {
        s_target_val = target_val;
    }
    __syncthreads();

    // --- Pass 2: compute sum of exp(z - max) ---
    float thread_sum = 0.0f;
    for (int v = threadIdx.x; v < local_vocab; v += kBlockThreads) {
        float z = __bfloat162float(row_logits[v]);
        thread_sum += expf(z - row_max);
    }

    float row_sum = block_reduce_sum(thread_sum, smem);

    // Write outputs (thread 0 only).
    if (threadIdx.x == 0) {
        local_max[row]          = row_max;
        // Store as log(sum_exp) + local_max  so that global LSE can be
        // computed via log-sum-exp of per-partition values.
        local_lse[row]          = logf(row_sum) + row_max;
        local_target_logit[row] = s_target_val;
        target_is_local[row]    = owns_target ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Backward / loss-combine kernel
// ---------------------------------------------------------------------------
// After an allreduce the host provides the *global* log-sum-exp for each
// row.  This kernel computes:
//   loss[row] = global_lse[row] - target_logit[row]  (only on owning tier)
//   dlogits[row, v] = softmax(logits[row,v]; global) - 1{v == target}
//
// One block per batch row.  Iterates over local_vocab.

template <int SmVer>
__global__ void __launch_bounds__(kBlockThreads)
fused_ce_backward_kernel(const __nv_bfloat16* __restrict__ logits,           // [B, V_local]
                         const int64_t*       __restrict__ targets,          // [B]
                         const float*         __restrict__ global_lse,       // [B]
                         const float*         __restrict__ local_target_logit, // [B]
                         const float*         __restrict__ target_is_local,  // [B]
                         float*               __restrict__ losses,           // [B]
                         __nv_bfloat16*       __restrict__ dlogits,          // [B, V_local]
                         int                               local_vocab,
                         int                               vocab_offset,
                         float                             loss_scale)
{
    const int row = blockIdx.x;

    const __nv_bfloat16* row_logits  = logits  + (size_t)row * local_vocab;
    __nv_bfloat16*       row_dlogits = dlogits + (size_t)row * local_vocab;

    const float glse     = global_lse[row];
    const int64_t global_target = targets[row];
    const bool owns_target = (global_target >= vocab_offset) &&
                             (global_target <  vocab_offset + local_vocab);
    const int local_target_idx = owns_target ? (int)(global_target - vocab_offset) : -1;

    // Write scalar loss (only on the tier that owns the target).
    if (threadIdx.x == 0 && owns_target) {
        losses[row] = (glse - local_target_logit[row]) * loss_scale;
    } else if (threadIdx.x == 0 && target_is_local[row] == 0.0f) {
        // No tier owns it on this partition — write 0 so allreduce-sum works.
        losses[row] = 0.0f;
    }

    // Compute softmax gradient: p_v - 1{v == target}
    // p_v = exp(z_v - global_lse)
    for (int v = threadIdx.x; v < local_vocab; v += kBlockThreads) {
        float z   = __bfloat162float(row_logits[v]);
        float p   = expf(z - glse);
        float ind = (v == local_target_idx) ? 1.0f : 0.0f;
        float g   = (p - ind) * loss_scale;
        row_dlogits[v] = __float2bfloat16(g);
    }
}

// ---------------------------------------------------------------------------
// Vectorised backward kernel — processes 8 BF16 elements per thread iteration
// ---------------------------------------------------------------------------

template <int SmVer>
__global__ void __launch_bounds__(kBlockThreads)
fused_ce_backward_vec_kernel(const __nv_bfloat16* __restrict__ logits,
                             const int64_t*       __restrict__ targets,
                             const float*         __restrict__ global_lse,
                             const float*         __restrict__ local_target_logit,
                             const float*         __restrict__ target_is_local,
                             float*               __restrict__ losses,
                             __nv_bfloat16*       __restrict__ dlogits,
                             int                               local_vocab,
                             int                               vocab_offset,
                             float                             loss_scale)
{
    const int row = blockIdx.x;

    const __nv_bfloat16* row_logits  = logits  + (size_t)row * local_vocab;
    __nv_bfloat16*       row_dlogits = dlogits + (size_t)row * local_vocab;

    const float glse     = global_lse[row];
    const int64_t global_target = targets[row];
    const bool owns_target = (global_target >= vocab_offset) &&
                             (global_target <  vocab_offset + local_vocab);
    const int local_target_idx = owns_target ? (int)(global_target - vocab_offset) : -1;

    if (threadIdx.x == 0) {
        if (owns_target) {
            losses[row] = (glse - local_target_logit[row]) * loss_scale;
        } else {
            losses[row] = 0.0f;
        }
    }

    // Vectorised path: process kVecWidth BF16 elements at a time.
    const int vec_elems = local_vocab / kVecWidth;
    using VecType = uint4;  // 16 bytes = 8 × BF16

    const VecType* logits_vec  = reinterpret_cast<const VecType*>(row_logits);
    VecType*       dlogits_vec = reinterpret_cast<VecType*>(row_dlogits);

    for (int vi = threadIdx.x; vi < vec_elems; vi += kBlockThreads) {
        VecType data = logits_vec[vi];
        __nv_bfloat16* elems = reinterpret_cast<__nv_bfloat16*>(&data);
        __nv_bfloat16 out_elems[kVecWidth];

        const int base_v = vi * kVecWidth;
#pragma unroll
        for (int k = 0; k < kVecWidth; k++) {
            float z   = __bfloat162float(elems[k]);
            float p   = expf(z - glse);
            float ind = ((base_v + k) == local_target_idx) ? 1.0f : 0.0f;
            float g   = (p - ind) * loss_scale;
            out_elems[k] = __float2bfloat16(g);
        }

        VecType out;
        *reinterpret_cast<__nv_bfloat16(*)[kVecWidth]>(&out) =
            *reinterpret_cast<__nv_bfloat16(*)[kVecWidth]>(&out_elems);
        dlogits_vec[vi] = out;
    }

    // Handle remainder elements (local_vocab not divisible by kVecWidth).
    const int remainder_start = vec_elems * kVecWidth;
    for (int v = remainder_start + threadIdx.x; v < local_vocab; v += kBlockThreads) {
        float z   = __bfloat162float(row_logits[v]);
        float p   = expf(z - glse);
        float ind = (v == local_target_idx) ? 1.0f : 0.0f;
        float g   = (p - ind) * loss_scale;
        row_dlogits[v] = __float2bfloat16(g);
    }
}

// ---------------------------------------------------------------------------
// SM-dispatch wrapper: forward
// ---------------------------------------------------------------------------

void launch_fused_ce_forward(const __nv_bfloat16* logits,
                             const int64_t*       targets,
                             float*               local_max,
                             float*               local_lse,
                             float*               local_target_logit,
                             float*               target_is_local,
                             int                  batch,
                             int                  local_vocab,
                             int                  vocab_offset,
                             int                  sm_version,
                             cudaStream_t         stream)
{
    dim3 grid(batch);
    dim3 block(kBlockThreads);

    switch (sm_version) {
    case 90:
        fused_ce_forward_kernel<90><<<grid, block, 0, stream>>>(
            logits, targets, local_max, local_lse,
            local_target_logit, target_is_local,
            local_vocab, vocab_offset);
        break;
    case 120:
        fused_ce_forward_kernel<120><<<grid, block, 0, stream>>>(
            logits, targets, local_max, local_lse,
            local_target_logit, target_is_local,
            local_vocab, vocab_offset);
        break;
    default:  // SM 8.6 and others
        fused_ce_forward_kernel<86><<<grid, block, 0, stream>>>(
            logits, targets, local_max, local_lse,
            local_target_logit, target_is_local,
            local_vocab, vocab_offset);
        break;
    }
}

// ---------------------------------------------------------------------------
// SM-dispatch wrapper: backward / loss-combine
// ---------------------------------------------------------------------------

void launch_fused_ce_backward(const __nv_bfloat16* logits,
                              const int64_t*       targets,
                              const float*         global_lse,
                              const float*         local_target_logit,
                              const float*         target_is_local,
                              float*               losses,
                              __nv_bfloat16*       dlogits,
                              int                  batch,
                              int                  local_vocab,
                              int                  vocab_offset,
                              float                loss_scale,
                              int                  sm_version,
                              cudaStream_t         stream)
{
    dim3 grid(batch);
    dim3 block(kBlockThreads);

    // Use vectorised kernel when local_vocab is large enough.
    const bool use_vec = (local_vocab >= kVecWidth * kBlockThreads);

    if (use_vec) {
        switch (sm_version) {
        case 90:
            fused_ce_backward_vec_kernel<90><<<grid, block, 0, stream>>>(
                logits, targets, global_lse, local_target_logit,
                target_is_local, losses, dlogits,
                local_vocab, vocab_offset, loss_scale);
            break;
        case 120:
            fused_ce_backward_vec_kernel<120><<<grid, block, 0, stream>>>(
                logits, targets, global_lse, local_target_logit,
                target_is_local, losses, dlogits,
                local_vocab, vocab_offset, loss_scale);
            break;
        default:
            fused_ce_backward_vec_kernel<86><<<grid, block, 0, stream>>>(
                logits, targets, global_lse, local_target_logit,
                target_is_local, losses, dlogits,
                local_vocab, vocab_offset, loss_scale);
            break;
        }
    } else {
        switch (sm_version) {
        case 90:
            fused_ce_backward_kernel<90><<<grid, block, 0, stream>>>(
                logits, targets, global_lse, local_target_logit,
                target_is_local, losses, dlogits,
                local_vocab, vocab_offset, loss_scale);
            break;
        case 120:
            fused_ce_backward_kernel<120><<<grid, block, 0, stream>>>(
                logits, targets, global_lse, local_target_logit,
                target_is_local, losses, dlogits,
                local_vocab, vocab_offset, loss_scale);
            break;
        default:
            fused_ce_backward_kernel<86><<<grid, block, 0, stream>>>(
                logits, targets, global_lse, local_target_logit,
                target_is_local, losses, dlogits,
                local_vocab, vocab_offset, loss_scale);
            break;
        }
    }
}
