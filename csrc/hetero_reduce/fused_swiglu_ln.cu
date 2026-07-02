// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_swiglu_ln.cu
 *
 * Fused SwiGLU activation + RMS LayerNorm kernel.
 *
 * For each row i of [batch × hidden]:
 *   swiglu_i[j] = gate_proj_i[j] * sigmoid(gate_proj_i[j]) * up_proj_i[j]
 *   output_i[j] = swiglu_i[j] * ln_weight[j] / rms(swiglu_i)
 *
 * RMS = sqrt( mean(x^2) + eps )
 *
 * Kernel design
 * -------------
 * One CTA per row.  Threads collectively load both projections in BF16,
 * compute SwiGLU in FP32, accumulate squared-sum for RMS (warp + block
 * reductions), then normalise and store BF16.
 *
 * Each thread handles kVecWidth=8 BF16 elements per iteration.
 * Shared memory holds per-warp partial sums for the block reduction.
 *
 * SM specialisations (compile-time template parameter SmVer):
 *   SmVer == 86  → A6000: kBlockSize=256, __launch_bounds__(256, 2)
 *   SmVer == 90  → H100 : kBlockSize=256, __launch_bounds__(256, 4) (larger L2)
 *   SmVer == 120 → Blackwell: kBlockSize=512, double-width warps hint
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

static constexpr int kLNVecWidth  = 8;   // BF16 elements per thread per step
static constexpr int kMaxWarps    = 16;  // max warps = kBlockSize / 32

// ---------------------------------------------------------------------------
// Fast sigmoid for scalar float (used in SwiGLU).
// ---------------------------------------------------------------------------
DS_D_INLINE float fast_sigmoid(float x)
{
    return 1.f / (1.f + __expf(-x));
}

// ---------------------------------------------------------------------------
// Warp-level reduction (sum).
// ---------------------------------------------------------------------------
DS_D_INLINE float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int offset = hw_warp_size / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Kernel template — SmVer selects occupancy hints.
// ---------------------------------------------------------------------------

template <int SmVer, int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, (SmVer >= 120) ? 4 : (SmVer >= 90) ? 4 : 2)
fused_swiglu_ln_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    const float* __restrict__         ln_weight,
    int hidden,
    float eps)
{
    // Each block handles one row.
    const int row    = blockIdx.x;
    const int lane   = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;
    const int n_warps = kBlockSize / hw_warp_size;

    const __nv_bfloat16* g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* o_row = output    + (size_t)row * hidden;

    // Shared mem: per-warp squared sums.
    __shared__ float smem_sq[kMaxWarps];

    float thread_sq_sum = 0.f;

    // -----------------------------------------------------------------------
    // Pass 1: compute SwiGLU into registers, accumulate squared sum.
    //         We process kLNVecWidth elements per thread per iteration, but
    //         we only have register space to store partial results per warp;
    //         the full row is streamed twice (pass 1 = sq-sum, pass 2 = store).
    //
    //         For hidden sizes that fit entirely in registers (≤ 8*kBlockSize
    //         elements) this would be one pass. We use a two-pass approach for
    //         generality (hidden can be 4096+).
    // -----------------------------------------------------------------------
    for (int col = threadIdx.x * kLNVecWidth; col < hidden; col += kBlockSize * kLNVecWidth) {
        // Load gate and up projections as uint4 (128 bits = 8 × BF16).
        uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
        uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);

        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

        #pragma unroll
        for (int v = 0; v < kLNVecWidth; v++) {
            float gv = __bfloat162float(gp[v]);
            float uv = __bfloat162float(up[v]);
            float sw = gv * fast_sigmoid(gv) * uv;
            thread_sq_sum += sw * sw;
        }
    }

    // Warp reduction.
    thread_sq_sum = warp_reduce_sum(thread_sq_sum);
    if (lane == 0) smem_sq[warp_id] = thread_sq_sum;
    __syncthreads();

    // Block reduction over warp sums (done by the first warp).
    float block_sq = 0.f;
    if (threadIdx.x < n_warps) {
        block_sq = smem_sq[threadIdx.x];
    }
    if (warp_id == 0) {
        block_sq = warp_reduce_sum(block_sq);
    }
    // Broadcast RMS denominator to all threads.
    float rms_inv = 0.f;
    if (threadIdx.x == 0) {
        smem_sq[0] = rsqrtf(block_sq / (float)hidden + eps);
    }
    __syncthreads();
    rms_inv = smem_sq[0];

    // -----------------------------------------------------------------------
    // Pass 2: recompute SwiGLU (no extra DRAM — gate/up already in L2 cache),
    //         multiply by LN weight, normalise, store BF16.
    // -----------------------------------------------------------------------
    for (int col = threadIdx.x * kLNVecWidth; col < hidden; col += kBlockSize * kLNVecWidth) {
        uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
        uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);

        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

        __nv_bfloat16 out_buf[kLNVecWidth];

        #pragma unroll
        for (int v = 0; v < kLNVecWidth; v++) {
            float gv  = __bfloat162float(gp[v]);
            float uv  = __bfloat162float(up[v]);
            float sw  = gv * fast_sigmoid(gv) * uv;
            float w   = ln_weight[col + v];
            float res = sw * rms_inv * w;
            out_buf[v] = __float2bfloat16(res);
        }

        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

// ---------------------------------------------------------------------------
// Host-side dispatch
// ---------------------------------------------------------------------------

void launch_fused_swiglu_ln(__nv_bfloat16* output,
                             const __nv_bfloat16* gate_proj,
                             const __nv_bfloat16* up_proj,
                             const float* ln_weight,
                             int batch,
                             int hidden,
                             float eps,
                             int sm_version,
                             cudaStream_t stream)
{
    const int grid = batch;  // one CTA per row

    if (sm_version >= 120) {
        // Blackwell: use 512-thread blocks for wider occupancy.
        constexpr int kBS = 512;
        fused_swiglu_ln_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
            output, gate_proj, up_proj, ln_weight, hidden, eps);
    } else if (sm_version >= 90) {
        // H100
        constexpr int kBS = 256;
        fused_swiglu_ln_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
            output, gate_proj, up_proj, ln_weight, hidden, eps);
    } else {
        // A6000 (SM 8.6) and older
        constexpr int kBS = 256;
        fused_swiglu_ln_kernel<86, kBS><<<grid, kBS, 0, stream>>>(
            output, gate_proj, up_proj, ln_weight, hidden, eps);
    }
}
