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
 * reductions via cooperative groups), then normalise and store BF16.
 *
 * Each thread handles kVecWidth=8 BF16 elements per iteration.
 * Shared memory holds per-warp partial sums for the block reduction.
 *
 * SM specialisations (compile-time template parameter SmVer):
 *   SmVer == 86  → A6000: kBlockSize=256, __launch_bounds__(256, 2)
 *   SmVer == 90  → H100 : kBlockSize=256, __launch_bounds__(256, 4) (larger L2)
 *   SmVer == 120 → Blackwell: kBlockSize=512, higher occupancy for wider SMs
 *
 * Cooperative groups
 * ------------------
 * All warp-level reductions use cooperative_groups::tiled_partition<32>
 * and cg::reduce() instead of raw __shfl_down_sync.  This gives the
 * compiler freedom to emit native warp-reduce instructions on Blackwell
 * and correct masks on all SM versions.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

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
// Warp-level reduction (sum) using cooperative groups.
// ---------------------------------------------------------------------------
DS_D_INLINE float cg_warp_reduce_sum(cg::thread_block_tile<32>& warp, float val)
{
    return cg::reduce(warp, val, cg::plus<float>());
}

// ---------------------------------------------------------------------------
// Vectorised 128-bit load helper for gate + up projections.
// Returns SwiGLU activation values in FP32 and accumulates squared sum.
// ---------------------------------------------------------------------------
DS_D_INLINE void load_and_swiglu(
    const __nv_bfloat16* __restrict__ g_row,
    const __nv_bfloat16* __restrict__ u_row,
    int col,
    float swiglu_vals[kLNVecWidth],
    float& sq_sum)
{
    uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
    uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);

    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
    const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

    #pragma unroll
    for (int v = 0; v < kLNVecWidth; v++) {
        float gv = __bfloat162float(gp[v]);
        float uv = __bfloat162float(up[v]);
        float sw = gv * fast_sigmoid(gv) * uv;
        swiglu_vals[v] = sw;
        sq_sum += sw * sw;
    }
}

// ---------------------------------------------------------------------------
// Kernel template — SmVer selects occupancy hints and block dimensions.
//
// SM 86  (A6000):    256 threads, 2 blocks/SM — conservative register usage
// SM 90  (H100):     256 threads, 4 blocks/SM — exploit larger L2 & SM count
// SM 120 (Blackwell):512 threads, 4 blocks/SM — wider SMs, more registers
// ---------------------------------------------------------------------------

template <int SmVer, int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
fused_swiglu_ln_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    const float* __restrict__         ln_weight,
    int hidden,
    float eps)
{
    // Each block handles one row.
    cg::thread_block blk = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(blk);

    const int row     = blockIdx.x;
    const int lane    = warp.thread_rank();
    const int warp_id = threadIdx.x / hw_warp_size;
    const int n_warps = kBlockSize / hw_warp_size;

    const __nv_bfloat16* g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* o_row = output    + (size_t)row * hidden;

    // Shared mem: per-warp squared sums for block reduction.
    __shared__ float smem_sq[kMaxWarps];

    float thread_sq_sum = 0.f;

    // -------------------------------------------------------------------
    // Pass 1: compute SwiGLU, accumulate squared sum for RMS.
    //
    // Two-pass approach for generality: hidden can be 4096+ and we cannot
    // hold the entire row in registers.  Pass 1 streams through to compute
    // the RMS denominator; pass 2 re-reads (from L2 cache) and normalises.
    // -------------------------------------------------------------------
    for (int col = threadIdx.x * kLNVecWidth; col < hidden;
         col += kBlockSize * kLNVecWidth) {
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

    // Warp reduction via cooperative groups.
    thread_sq_sum = cg_warp_reduce_sum(warp, thread_sq_sum);
    if (lane == 0) smem_sq[warp_id] = thread_sq_sum;
    blk.sync();

    // Block reduction over warp sums (first warp only).
    float block_sq = 0.f;
    if (threadIdx.x < n_warps) {
        block_sq = smem_sq[threadIdx.x];
    }
    if (warp_id == 0) {
        block_sq = cg_warp_reduce_sum(warp, block_sq);
    }

    // Broadcast RMS inverse to all threads via shared memory.
    if (threadIdx.x == 0) {
        smem_sq[0] = rsqrtf(block_sq / (float)hidden + eps);
    }
    blk.sync();
    float rms_inv = smem_sq[0];

    // -------------------------------------------------------------------
    // Pass 2: recompute SwiGLU (gate/up in L2 cache from pass 1),
    //         multiply by LN weight, normalise, store BF16.
    // -------------------------------------------------------------------
    for (int col = threadIdx.x * kLNVecWidth; col < hidden;
         col += kBlockSize * kLNVecWidth) {
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
        // Blackwell: use 512-thread blocks for wider SMs.
        constexpr int kBS = 512;
        fused_swiglu_ln_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
            output, gate_proj, up_proj, ln_weight, hidden, eps);
    } else if (sm_version >= 90) {
        // H100: 256-thread blocks, 4 CTAs per SM.
        constexpr int kBS = 256;
        fused_swiglu_ln_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
            output, gate_proj, up_proj, ln_weight, hidden, eps);
    } else {
        // A6000 (SM 8.6) and older: 256-thread blocks, 2 CTAs per SM.
        constexpr int kBS = 256;
        fused_swiglu_ln_kernel<86, kBS><<<grid, kBS, 0, stream>>>(
            output, gate_proj, up_proj, ln_weight, hidden, eps);
    }
}
